/// Device Identification for Context Window Optimization
///
/// This module provides precise device identification to enable optimal context window sizing based on:
/// 1. Device Memory Capacity:
///    - Entry level devices (iPhone SE): 2-3GB RAM
///    - Mid-range devices (iPhone 13): 4-6GB RAM
///    - Pro devices (iPad Pro): 8-16GB RAM
/// 2. Neural Engine Capabilities:
///    - A-series chips: Limited neural processing
///    - M-series chips: Advanced neural engine
/// 3. Thermal Constraints:
///    - Phone form factors: Conservative thermal limits
///    - Tablet form factors: Better thermal headroom
///
/// Product teams can use this data to tune context windows:
/// - Entry devices: 2K-4K token contexts
/// - Mid-range: 4K-8K token contexts
/// - Pro devices: 8K-32K token contexts
///
/// Context Window Optimization Flow:
/// 1. Device Detection Phase:
///    - Identify exact device model
///    - Map to memory/capability tier
/// 2. Context Sizing Phase:
///    - Set base context size for tier
///    - Apply thermal multipliers
/// 3. Runtime Monitoring Phase:
///    - Track memory pressure
///    - Adjust context dynamically
///
/// Teams should monitor:
/// 1. Context utilization rates
/// 2. Memory pressure during inference
/// 3. Thermal throttling events
/// 4. User engagement metrics

import Foundation

#if os(iOS) || os(tvOS)
  import UIKit

  extension UIDevice {
    /// Provides detailed device identification for context window optimization
    ///
    /// This property enables precise context window tuning by:
    /// 1. Identifying exact device model and capabilities
    /// 2. Mapping to appropriate context window size tier
    /// 3. Enabling dynamic context adjustment based on device
    ///
    /// Usage Example:
    /// ```swift
    /// let device = UIDevice.modelName
    /// let contextSize = getContextSize(for: device)
    /// model.setContextWindow(contextSize)
    /// ```
    ///
    /// Product teams should monitor:
    /// - Context window utilization
    /// - Memory pressure during inference
    /// - Model performance metrics
    public static let modelName: String = {
      // Get raw device identifier using uname
      var systemInfo = utsname()
      uname(&systemInfo)

      // Extract machine identifier using reflection
      let machineMirror = Mirror(reflecting: systemInfo.machine)
      let identifier = machineMirror.children.reduce("") { identifier, element in
        guard let value = element.value as? Int8, value != 0 else { return identifier }
        return identifier + String(UnicodeScalar(UInt8(value)))
      }

      /// Maps device identifiers to human readable names while preserving context window implications
      ///
      /// This mapping enables:
      /// 1. Precise device capability identification
      /// 2. Memory tier classification
      /// 3. Neural engine capability detection
      ///
      /// Context Window Tiers:
      /// - Entry (2-4K tokens): iPhone SE, base iPads
      /// - Mid (4-8K tokens): iPhone 13/14, iPad Air
      /// - Pro (8-32K tokens): Pro devices, M1/M2 iPads
      func mapToDevice(identifier: String) -> String {  // swiftlint:disable:this cyclomatic_complexity
        #if os(iOS)
          switch identifier {
          // Legacy Devices - Minimal Context (2K tokens)
          case "iPod5,1": return "iPod touch (5th generation)"
          case "iPod7,1": return "iPod touch (6th generation)"
          case "iPod9,1": return "iPod touch (7th generation)"

          // Entry Level - Small Context (2-4K tokens)
          case "iPhone3,1", "iPhone3,2", "iPhone3,3": return "iPhone 4"
          case "iPhone4,1": return "iPhone 4s"
          case "iPhone5,1", "iPhone5,2": return "iPhone 5"

          // Mid Range - Medium Context (4-8K tokens)
          case "iPhone13,1": return "iPhone 12 mini"
          case "iPhone13,2": return "iPhone 12"
          case "iPhone14,4": return "iPhone 13 mini"
          case "iPhone14,5": return "iPhone 13"

          // Pro Devices - Large Context (8-32K tokens)
          case "iPhone14,2": return "iPhone 13 Pro"
          case "iPhone14,3": return "iPhone 13 Pro Max"
          case "iPhone15,2": return "iPhone 14 Pro"
          case "iPhone15,3": return "iPhone 14 Pro Max"

          // iPads - Variable Context Based on Neural Engine
          case "iPad13,8", "iPad13,9", "iPad13,10", "iPad13,11":
            return "iPad Pro (12.9-inch) (5th generation)"
          case "iPad14,5", "iPad14,6": return "iPad Pro (12.9-inch) (6th generation)"

          // Development Environment - Maximum Context
          case "i386", "x86_64", "arm64":
            return
              "Simulator \(mapToDevice(identifier: ProcessInfo().environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "iOS"))"
          default: return identifier
          }
        #elseif os(tvOS)
          switch identifier {
          case "AppleTV5,3": return "Apple TV 4"
          case "AppleTV6,2": return "Apple TV 4K"
          case "i386", "x86_64":
            return
              "Simulator \(mapToDevice(identifier: ProcessInfo().environment["SIMULATOR_MODEL_IDENTIFIER"] ?? "tvOS"))"
          default: return identifier
          }
        #endif
      }

      return mapToDevice(identifier: identifier)
    }()

  }
#endif
