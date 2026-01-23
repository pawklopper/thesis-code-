#include <ros/ros.h>
#include <geometry_msgs/WrenchStamped.h>

#include <atomic>
#include <thread>
#include <mutex>
#include <cmath>

#include "dhdc.h"

// ============================================================
// ALL CONTROL PARAMETERS (moved to the top of the file)
// ============================================================

// Rates
static constexpr double SERVO_HZ_DEFAULT   = 4000.0;
static constexpr double PUBLISH_HZ_DEFAULT = 500.0;

// Linear gains (XY)
static constexpr double LINEAR_STIFFNESS_XY_DEFAULT = 550.0; // N/m
static constexpr double LINEAR_VISCOSITY_XY_DEFAULT = 50.0;   // N/(m/s)

// Linear gains (Z) - anisotropic
static constexpr double LINEAR_STIFFNESS_Z_DEFAULT  = 2000.0; // N/m
static constexpr double LINEAR_VISCOSITY_Z_DEFAULT  = 20.0;   // N/(m/s)

// Angular damping
static constexpr double ANGULAR_VISCOSITY_DEFAULT   = 0.1;    // Nm/(rad/s)

// Gripper (hold at startup angle by default)
static constexpr double GRIPPER_STIFFNESS_DEFAULT   = 10.0;   // N/rad
static constexpr double GRIPPER_VISCOSITY_DEFAULT   = 20.0;   // N/(rad/s)  (reuse-like magnitude)

// Force limits (decoupled)
static constexpr double MAX_FORCE_XY_NORM_DEFAULT   = 20.0;   // N (norm clamp on XY)
static constexpr double MAX_FORCE_Z_ABS_DEFAULT     = 60.0;   // N (abs clamp on Z)

// Startup “slow move to origin” ramp
static constexpr double RAMP_TIME_DEFAULT           = 2.0;    // s (time to move equilibrium to target)
static constexpr double RAMP_MAX_FORCE_XY_DEFAULT   = 10.0;   // N (gentler during ramp)
static constexpr double RAMP_MAX_FORCE_Z_DEFAULT    = 10.0;   // N

// Ramp target (“new origin”) in dhdGetPosition() frame
static constexpr bool   RAMP_TARGET_USE_CURRENT_DEFAULT = false; // if true: capture target at ramp start
static constexpr double RAMP_TARGET_X_DEFAULT           = 0.015;   // m 
 static constexpr double RAMP_TARGET_Y_DEFAULT           = -0.02;   // m
static constexpr double RAMP_TARGET_Z_DEFAULT           = 0.0;   // mcatk

// ============================================================

struct SharedWrench
{
  double fx{0}, fy{0}, fz{0};
  double tx{0}, ty{0}, tz{0};
  double grip_force{0};
  double px{0}, py{0}, pz{0};
  double vx{0}, vy{0}, vz{0};
};

static std::mutex g_mtx;
static SharedWrench g_state;
static std::atomic<bool> g_running{true};

static inline void clamp2(double &x, double &y, double max_norm)
{
  const double n = std::sqrt(x*x + y*y);
  if (n > max_norm && n > 1e-12) {
    const double s = max_norm / n;
    x *= s; y *= s;
  }
}

static inline void clampAbs(double &v, double max_abs)
{
  if (v >  max_abs) v =  max_abs;
  if (v < -max_abs) v = -max_abs;
}

/**
 * @brief Startup phase: slowly moves the controller equilibrium from the
 *        current position to a target position (x_t,y_t,z_t) over ramp_time seconds.
 *
 * Implementation: interpolate desired position from p0 -> p_target and apply the
 * same spring-damper law, but with gentler clamps during the ramp.
 */
static void rampToOrigin(
    double ramp_time_s,
    double ramp_max_force_xy,
    double ramp_max_force_z,
    double linear_stiffness_xy,
    double linear_viscosity_xy,
    double linear_stiffness_z,
    double linear_viscosity_z,
    double angular_viscosity,
    double gripper_stiffness,
    double gripper_viscosity,
    double gripper_target_angle,
    double ramp_target_x,
    double ramp_target_y,
    double ramp_target_z)
{
  if (ramp_time_s <= 1e-6) return;

  double p0x=0, p0y=0, p0z=0;
  if (dhdGetPosition(&p0x, &p0y, &p0z) < 0) {
    ROS_WARN("rampToOrigin: failed to read initial position; skipping ramp.");
    return;
  }

  const double t_start = dhdGetTime();
  const double t_end   = t_start + ramp_time_s;

  while (g_running.load(std::memory_order_relaxed) && ros::ok())
  {
    const double t_now = dhdGetTime();
    if (t_now >= t_end) break;

    const double alpha = (t_now - t_start) / ramp_time_s; // 0 -> 1

    // Desired position moves linearly from current position to target
    const double x_d = (1.0 - alpha) * p0x + alpha * ramp_target_x;
    const double y_d = (1.0 - alpha) * p0y + alpha * ramp_target_y;
    const double z_d = (1.0 - alpha) * p0z + alpha * ramp_target_z;

    double px, py, pz;
    double vx, vy, vz;
    double wx, wy, wz;
    double pg, vg;

    if (dhdGetPosition(&px, &py, &pz) < 0) continue;
    if (dhdGetLinearVelocity(&vx, &vy, &vz) < 0) continue;
    dhdGetAngularVelocityRad(&wx, &wy, &wz);
    dhdGetGripperAngleRad(&pg);
    dhdGetGripperLinearVelocity(&vg);

    const double ex = x_d - px;
    const double ey = y_d - py;
    const double ez = z_d - pz;

    double fx = linear_stiffness_xy * ex - linear_viscosity_xy * vx;
    double fy = linear_stiffness_xy * ey - linear_viscosity_xy * vy;
    double fz = linear_stiffness_z  * ez - linear_viscosity_z  * vz;

    if (ramp_max_force_xy > 0.0) clamp2(fx, fy, ramp_max_force_xy);
    if (ramp_max_force_z  > 0.0) clampAbs(fz, ramp_max_force_z);

    const double tx = -angular_viscosity * wx;
    const double ty = -angular_viscosity * wy;
    const double tz = -angular_viscosity * wz;

    const double fg = gripper_stiffness * (gripper_target_angle - pg) - gripper_viscosity * vg;

    if (dhdSetForceAndTorqueAndGripperForce(fx, fy, fz, tx, ty, tz, fg) < DHD_NO_ERROR) {
      ROS_ERROR("rampToOrigin: dhdSetForce... failed: %s", dhdErrorGetLastStr());
      break;
    }

    // Keep the servo loop timing reasonable during ramp
    dhdSleep(1.0 / SERVO_HZ_DEFAULT);
  }
}

static void servoThread(



    // gains
    double linear_stiffness_xy,
    double linear_viscosity_xy,
    double linear_stiffness_z,
    double linear_viscosity_z,
    double angular_viscosity,
    double gripper_stiffness,
    double gripper_viscosity,

    // limits
    double max_force_xy_norm,
    double max_force_z_abs,

    // ramp
    double ramp_time_s,
    double ramp_max_force_xy,
    double ramp_max_force_z,
    bool   ramp_target_use_current,
    double ramp_target_x,
    double ramp_target_y,
    double ramp_target_z)



{
  if (dhdOpen() < 0) {
    ROS_ERROR("dhdOpen failed: %s", dhdErrorGetLastStr());
    g_running = false;
    return;
  }

  ROS_INFO("%s device detected", dhdGetSystemName());

  if (dhdEnableForce(DHD_ON) < 0) {
    ROS_ERROR("dhdEnableForce failed: %s", dhdErrorGetLastStr());
    dhdClose();
    g_running = false;
    return;
  }

  // Hold the gripper at its startup angle (no “setpoint parameter” anymore)
  double gripper_target_angle = 0.0;
  dhdGetGripperAngleRad(&gripper_target_angle);

  // Persistent equilibrium (used both for ramp target and steady-state setpoint)
  double eq_x = ramp_target_x;
  double eq_y = ramp_target_y;
  double eq_z = ramp_target_z;

  if (ramp_target_use_current) {
    double cx=0, cy=0, cz=0;
    if (dhdGetPosition(&cx, &cy, &cz) >= 0) {
      eq_x = cx; eq_y = cy; eq_z = cz;
    } else {
      ROS_WARN("Failed to capture current position for equilibrium; using provided ramp_target_(x,y,z).");
    }
  }

  ROS_INFO("Equilibrium target in dhdGetPosition() frame: (%.6f, %.6f, %.6f) m",
           eq_x, eq_y, eq_z);

  // Startup gentle move-to-target
  if (ramp_time_s > 1e-6) {
    ROS_INFO("Ramping equilibrium to target over %.2f s (gentle clamps XY=%.2f N, Z=%.2f N).",
             ramp_time_s, ramp_max_force_xy, ramp_max_force_z);
    rampToOrigin(ramp_time_s,
                 ramp_max_force_xy,
                 ramp_max_force_z,
                 linear_stiffness_xy,
                 linear_viscosity_xy,
                 linear_stiffness_z,
                 linear_viscosity_z,
                 angular_viscosity,
                 gripper_stiffness,
                 gripper_viscosity,
                 gripper_target_angle,
                 eq_x,
                 eq_y,
                 eq_z);
  }

  const double period = 1.0 / SERVO_HZ_DEFAULT;
  double t_next = dhdGetTime();

  // ============================
  // Servo rate instrumentation
  // ============================
  double last_report_t = dhdGetTime();
  uint64_t iters = 0;
  double min_dt = 1e9, max_dt = 0.0;
  double last_t = dhdGetTime();
  // ============================

  while (g_running.load(std::memory_order_relaxed) && ros::ok())
  {
    // ============================
    // Servo rate instrumentation
    // ============================
    const double now_t0 = dhdGetTime();
    const double dt0 = now_t0 - last_t;
    last_t = now_t0;

    if (dt0 > 0.0) {
      if (dt0 < min_dt) min_dt = dt0;
      if (dt0 > max_dt) max_dt = dt0;
    }

    iters++;

    // if (now_t0 - last_report_t >= 1.0) {
    //   const double elapsed = now_t0 - last_report_t;
    //   const double hz = static_cast<double>(iters) / elapsed;

    //   ROS_INFO("Servo rate: %.1f Hz | dt min/avg/max: %.6f / %.6f / %.6f s",
    //            hz,
    //            min_dt,
    //            elapsed / static_cast<double>(iters),
    //            max_dt);

    //   last_report_t = now_t0;
    //   iters = 0;
    //   min_dt = 1e9;
    //   max_dt = 0.0;
    // }
    // ============================

    t_next += period;
    const double t_now = dhdGetTime();
    const double to_sleep = t_next - t_now;
    if (to_sleep > 0.0) dhdSleep(to_sleep);

    double px, py, pz;
    double vx, vy, vz;
    double wx, wy, wz;
    double pg, vg;

    if (dhdGetPosition(&px, &py, &pz) < 0) continue;
    if (dhdGetLinearVelocity(&vx, &vy, &vz) < 0) continue;
    dhdGetAngularVelocityRad(&wx, &wy, &wz);
    dhdGetGripperAngleRad(&pg);
    dhdGetGripperLinearVelocity(&vg);

    // Desired is the persistent equilibrium target
    const double x_d = eq_x;
    const double y_d = eq_y;
    const double z_d = eq_z;

    // Errors in RAW device frame
    const double ex = x_d - px;
    const double ey = y_d - py;
    const double ez = z_d - pz;

    // Spring-damper (anisotropic Z)
    double fx = linear_stiffness_xy * ex - linear_viscosity_xy * vx;
    double fy = linear_stiffness_xy * ey - linear_viscosity_xy * vy;
    double fz = linear_stiffness_z  * ez - linear_viscosity_z  * vz;

    // Limits (decoupled to avoid Z scaling XY)
    if (max_force_xy_norm > 0.0) clamp2(fx, fy, max_force_xy_norm);
    if (max_force_z_abs  > 0.0) clampAbs(fz, max_force_z_abs);

    // Angular damping
    const double tx = -angular_viscosity * wx;
    const double ty = -angular_viscosity * wy;
    const double tz = -angular_viscosity * wz;

    // Gripper: hold at startup angle (no external setpoint)
    const double fg = gripper_stiffness * (gripper_target_angle - pg) - gripper_viscosity * vg;

    if (dhdSetForceAndTorqueAndGripperForce(fx, fy, fz, tx, ty, tz, fg) < DHD_NO_ERROR) {
      ROS_ERROR("dhdSetForce... failed: %s", dhdErrorGetLastStr());
      break;
    }

    {
      std::lock_guard<std::mutex> lk(g_mtx);
      g_state.fx = fx; g_state.fy = fy; g_state.fz = fz;
      g_state.tx = tx; g_state.ty = ty; g_state.tz = tz;
      g_state.grip_force = fg;
      g_state.px = px; g_state.py = py; g_state.pz = pz;
      g_state.vx = vx; g_state.vy = vy; g_state.vz = vz;
    }
  }

  dhdSetForceAndTorqueAndGripperForce(0,0,0, 0,0,0, 0);
  dhdClose();
  g_running = false;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sigma7_haptic_node");
  ros::NodeHandle nh("~");

  // Gains (XY)
  double linear_stiffness_xy = LINEAR_STIFFNESS_XY_DEFAULT;
  double linear_viscosity_xy = LINEAR_VISCOSITY_XY_DEFAULT;
  nh.param("linear_stiffness_xy", linear_stiffness_xy, linear_stiffness_xy);
  nh.param("linear_viscosity_xy", linear_viscosity_xy, linear_viscosity_xy);

  // Gains (Z)
  double linear_stiffness_z = LINEAR_STIFFNESS_Z_DEFAULT;
  double linear_viscosity_z = LINEAR_VISCOSITY_Z_DEFAULT;
  nh.param("linear_stiffness_z", linear_stiffness_z, linear_stiffness_z);
  nh.param("linear_viscosity_z", linear_viscosity_z, linear_viscosity_z);

  // Angular / gripper
  double angular_viscosity = ANGULAR_VISCOSITY_DEFAULT;
  double gripper_stiffness = GRIPPER_STIFFNESS_DEFAULT;
  double gripper_viscosity = GRIPPER_VISCOSITY_DEFAULT;
  nh.param("angular_viscosity", angular_viscosity, angular_viscosity);
  nh.param("gripper_stiffness", gripper_stiffness, gripper_stiffness);
  nh.param("gripper_viscosity", gripper_viscosity, gripper_viscosity);

  // Limits (decoupled)
  double max_force_xy_norm = MAX_FORCE_XY_NORM_DEFAULT;
  double max_force_z_abs   = MAX_FORCE_Z_ABS_DEFAULT;
  nh.param("max_force_xy_norm", max_force_xy_norm, max_force_xy_norm);
  nh.param("max_force_z_abs",   max_force_z_abs,   max_force_z_abs);

  // Ramp parameters
  double ramp_time_s       = RAMP_TIME_DEFAULT;
  double ramp_max_force_xy = RAMP_MAX_FORCE_XY_DEFAULT;
  double ramp_max_force_z  = RAMP_MAX_FORCE_Z_DEFAULT;
  nh.param("ramp_time_s",       ramp_time_s,       ramp_time_s);
  nh.param("ramp_max_force_xy", ramp_max_force_xy, ramp_max_force_xy);
  nh.param("ramp_max_force_z",  ramp_max_force_z,  ramp_max_force_z);

  // Ramp target (“new origin”)
  bool   ramp_target_use_current = RAMP_TARGET_USE_CURRENT_DEFAULT;
  double ramp_target_x = RAMP_TARGET_X_DEFAULT;
  double ramp_target_y = RAMP_TARGET_Y_DEFAULT;
  double ramp_target_z = RAMP_TARGET_Z_DEFAULT;
  nh.param("ramp_target_use_current", ramp_target_use_current, ramp_target_use_current);
  nh.param("ramp_target_x",           ramp_target_x,           ramp_target_x);
  nh.param("ramp_target_y",           ramp_target_y,           ramp_target_y);
  nh.param("ramp_target_z",           ramp_target_z,           ramp_target_z);

  ros::Publisher pub = nh.advertise<geometry_msgs::WrenchStamped>("/sigma7/wrench", 10);

  std::thread servo(
      servoThread,
      linear_stiffness_xy,
      linear_viscosity_xy,
      linear_stiffness_z,
      linear_viscosity_z,
      angular_viscosity,
      gripper_stiffness,
      gripper_viscosity,
      max_force_xy_norm,
      max_force_z_abs,
      ramp_time_s,
      ramp_max_force_xy,
      ramp_max_force_z,
      ramp_target_use_current,
      ramp_target_x,
      ramp_target_y,
      ramp_target_z);

  ros::Rate r(PUBLISH_HZ_DEFAULT);
  while (ros::ok() && g_running.load())
  {
    SharedWrench s;
    {
      std::lock_guard<std::mutex> lk(g_mtx);
      s = g_state;
    }

    geometry_msgs::WrenchStamped msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "sigma7_base";
    msg.wrench.force.x  = s.fx;
    msg.wrench.force.y  = s.fy;
    msg.wrench.force.z  = s.fz;
    msg.wrench.torque.x = s.tx;
    msg.wrench.torque.y = s.ty;
    msg.wrench.torque.z = s.tz;

    pub.publish(msg);

    ros::spinOnce();
    r.sleep();
  }

  g_running = false;
  if (servo.joinable()) servo.join();
  return 0;
}
