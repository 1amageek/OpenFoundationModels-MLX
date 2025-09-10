import Foundation
import MLXLMCommon
import MLXLLM
import Hub

/// Model loader that handles downloading and loading models with progress reporting.
/// This class is completely independent from the inference layer (MLXLanguageModel).
public final class ModelLoader {
    
    // MARK: - Properties
    
    private let hubApi: HubApi
    private var modelCache: [String: ModelContainer] = [:]
    private let cacheQueue = DispatchQueue(label: "com.openFoundationModels.modelLoader.cache")
    
    // MARK: - Initialization
    
    /// Initialize a new model loader
    /// - Parameter hubApi: The Hub API to use for downloading models (defaults to HubApi())
    public init(hubApi: HubApi = HubApi()) {
        self.hubApi = hubApi
    }
    
    // MARK: - Public Methods
    
    /// Load a model with optional progress reporting
    /// - Parameters:
    ///   - modelID: The HuggingFace model ID (e.g., "mlx-community/llama-3-8b")
    ///   - progress: Optional Progress object for monitoring load progress
    /// - Returns: A loaded ModelContainer ready for inference
    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        
        // Check cache first
        if let cached = getCachedModel(modelID) {
            // If we have a progress object, mark it as complete
            progress?.completedUnitCount = progress?.totalUnitCount ?? 100
            progress?.localizedDescription = NSLocalizedString("Model loaded from cache", comment: "")
            return cached
        }
        
        // Set up progress if provided
        let loadingProgress = progress ?? Progress(totalUnitCount: 100)
        loadingProgress.localizedDescription = NSLocalizedString("Loading model...", comment: "")
        
        // Create model configuration
        let config = ModelConfiguration(id: modelID)
        
        // Load the model container
        let container = try await MLXLMCommon.loadModelContainer(
            hub: hubApi,
            configuration: config
        ) { hubProgress in
            // Map hub progress to our progress
            let fraction = hubProgress.fractionCompleted
            loadingProgress.completedUnitCount = Int64(fraction * Double(loadingProgress.totalUnitCount))
            
            // Update localized description if available
            if let description = hubProgress.localizedAdditionalDescription {
                loadingProgress.localizedAdditionalDescription = description
            }
        }
        
        // Cache the loaded model
        setCachedModel(container, for: modelID)
        
        // Mark progress as complete
        loadingProgress.completedUnitCount = loadingProgress.totalUnitCount
        loadingProgress.localizedDescription = NSLocalizedString("Model ready", comment: "")
        
        return container
    }
    
    /// Download a model without loading it into memory
    /// - Parameters:
    ///   - modelID: The HuggingFace model ID
    ///   - progress: Optional Progress object for monitoring download progress
    public func downloadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws {
        // This would download the model files without loading them
        // For now, we'll just use the load mechanism
        // In a full implementation, this would use Hub API directly to download files
        _ = try await loadModel(modelID, progress: progress)
    }
    
    /// Load a model from a local directory
    /// - Parameter path: The local path to the model directory
    /// - Returns: A loaded ModelContainer
    public func loadLocalModel(from path: URL) async throws -> ModelContainer {
        // Create a configuration for local loading
        let config = ModelConfiguration(directory: path)
        
        // Load from local path
        let container = try await MLXLMCommon.loadModelContainer(
            hub: hubApi,
            configuration: config
        ) { _ in
            // Local loading doesn't report progress
        }
        
        return container
    }
    
    /// Get a list of cached model IDs
    /// - Returns: Array of model IDs that are currently cached
    public func cachedModels() -> [String] {
        cacheQueue.sync {
            Array(modelCache.keys)
        }
    }
    
    /// Clear all cached models to free memory
    public func clearCache() {
        cacheQueue.sync {
            modelCache.removeAll()
        }
    }
    
    /// Clear a specific model from cache
    /// - Parameter modelID: The model ID to remove from cache
    public func clearCache(for modelID: String) {
        cacheQueue.sync {
            modelCache.removeValue(forKey: modelID)
        }
    }
    
    /// Check if a model is cached
    /// - Parameter modelID: The model ID to check
    /// - Returns: true if the model is cached
    public func isCached(_ modelID: String) -> Bool {
        cacheQueue.sync {
            modelCache[modelID] != nil
        }
    }
    
    // MARK: - Private Methods
    
    private func getCachedModel(_ modelID: String) -> ModelContainer? {
        cacheQueue.sync {
            modelCache[modelID]
        }
    }
    
    private func setCachedModel(_ container: ModelContainer, for modelID: String) {
        cacheQueue.sync {
            modelCache[modelID] = container
        }
    }
}