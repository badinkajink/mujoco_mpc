#include "mjpc/deploy/deploy_net.h"

#include <cstring>

// POSIX networking -- auto-pin the wired robot-subnet NIC.
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

namespace h12deploy {

std::string AutoDetectRobotInterface() {
  const char kRobotPrefix[] = "192.168.123.";
  std::string result;
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) == -1) return result;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family != AF_INET) continue;
    char host[INET_ADDRSTRLEN] = {0};
    const void* sin_addr =
        &reinterpret_cast<const struct sockaddr_in*>(ifa->ifa_addr)->sin_addr;
    if (inet_ntop(AF_INET, sin_addr, host, sizeof(host)) == nullptr) continue;
    if (std::strncmp(host, kRobotPrefix, sizeof(kRobotPrefix) - 1) == 0) {
      result = ifa->ifa_name;
      break;
    }
  }
  freeifaddrs(ifaddr);
  return result;
}

uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1u << 31;
    uint32_t data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }
  return CRC32;
}

}  // namespace h12deploy
