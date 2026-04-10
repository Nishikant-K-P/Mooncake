// Copyright 2024 KVCache.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PCI_TOPOLOGY_H
#define PCI_TOPOLOGY_H

#include <cctype>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mooncake {

enum class PciPathType {
    PIX = 0,   // Same PCI switch
    PXB = 1,   // Crossing PCI bridges
    PHB = 2,   // Crossing PCH/host bridge
    NODE = 3,  // Same NUMA node, no common PCI ancestor
    SYS = 4,   // Cross-NUMA
    DIS = 5,   // Disconnected
};

struct InfinibandDevice {
    std::string name;
    std::string pci_bus_id;
    int numa_node;
    double
        pci_bw_gbps;  // PCIe bandwidth in GT/s * width (e.g. 1024 for Gen5 x16)
};

constexpr size_t kMaxPciDepth = 32;

// Normalize a PCI bus ID: lowercase and strip leading zeros from the domain
// so that CUDA's 8-digit format (e.g. "00000008:06:00.0") matches sysfs's
// 4-digit format (e.g. "0008:06:00.0").
inline std::string normalizePciBusId(const std::string &pci_bus_id) {
    std::string normalized = pci_bus_id;
    for (char &ch : normalized) {
        ch = std::tolower(static_cast<unsigned char>(ch));
    }
    auto first_colon = normalized.find(':');
    if (first_colon != std::string::npos && first_colon > 4) {
        size_t leading = first_colon - 4;
        if (normalized.substr(0, leading) == std::string(leading, '0')) {
            normalized.erase(0, leading);
        }
    }
    return normalized;
}

// Classify the PCI path between a GPU and a NIC given their pre-built
// ancestor chains and NUMA node information.  This is the pure-logic core
// that can be unit-tested without sysfs access.
inline std::pair<PciPathType, int> classifyGpuNicPath(
    const std::vector<std::string> &gpu_ancestor_chain,
    const std::unordered_set<std::string> &gpu_ancestors, int gpu_numa_node,
    int nic_numa_node, const std::vector<std::string> &nic_ancestor_chain) {
    if (gpu_numa_node >= 0 && nic_numa_node >= 0 &&
        gpu_numa_node != nic_numa_node) {
        return {PciPathType::SYS, -1};
    }

    int nic_hops = 0;
    for (const auto &ancestor : nic_ancestor_chain) {
        if (gpu_ancestors.count(ancestor)) {
            int gpu_hops = 0;
            for (const auto &gpu_ancestor : gpu_ancestor_chain) {
                if (gpu_ancestor == ancestor) {
                    break;
                }
                gpu_hops++;
            }

            int total_hops = gpu_hops + nic_hops;
            if (total_hops <= 2) {
                return {PciPathType::PIX, total_hops};
            }
            if (total_hops <= 4) {
                return {PciPathType::PXB, total_hops};
            }
            return {PciPathType::PHB, total_hops};
        }
        nic_hops++;
    }

    if (gpu_numa_node >= 0 && gpu_numa_node == nic_numa_node) {
        // All same-NUMA NICs with no common PCI ancestor are equally close;
        // chain length is not a meaningful proximity metric here.  Return a
        // fixed hop count so the selection logic treats them identically.
        return {PciPathType::NODE, 0};
    }
    return {PciPathType::SYS, -1};
}

}  // namespace mooncake

#endif  // PCI_TOPOLOGY_H
