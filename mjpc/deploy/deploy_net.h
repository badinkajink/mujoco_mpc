// Deploy networking helpers (stage 3a of the 2026-07-18 reorg; formerly
// internal to deploy_common.cc).

#ifndef MJPC_DEPLOY_DEPLOY_NET_H_
#define MJPC_DEPLOY_DEPLOY_NET_H_

#include <cstdint>
#include <string>

namespace h12deploy {

// Auto-detect the interface holding a 192.168.123.x address (the wired H1-2
// robot subnet), so an EMPTY --network_interface binds the robot link instead
// of CycloneDDS autodetermine grabbing WiFi/Tailscale -- the trap that silently
// makes the node hear the same-host twin but never the real robot. Mirrors
// dds_tools/dds_topic_check.py. Returns "" when no robot-subnet NIC is present
// (-> caller keeps autodetermine/loopback, the right default for the twin).
std::string AutoDetectRobotInterface();

// Unitree LowCmd CRC (matches example/h1/low_level + unitree_sdk2py.utils.crc).
uint32_t Crc32Core(uint32_t* ptr, uint32_t len);

}  // namespace h12deploy

#endif  // MJPC_DEPLOY_DEPLOY_NET_H_
