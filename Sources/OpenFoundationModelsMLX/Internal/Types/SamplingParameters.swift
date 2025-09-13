//
//  SamplingParameters.swift
//  OpenFoundationModels-MLX
//
//  Created by 1amageek on 2025/09/13.
//

import Foundation
import MLXLMCommon

// Internal types for parameter conversion
// These minimal types support the internal engine design while keeping
// the OpenFoundationModels LanguageModel API intact.

// SamplingParameters is still used for parameter conversion between
// GenerationOptions and GenerateParameters
struct SamplingParameters: Codable, Sendable {
    var temperature: Double?
    var topP: Double?
    var topK: Int?
    var maxTokens: Int?
    var stop: [String]?
    var seed: Int?
}
