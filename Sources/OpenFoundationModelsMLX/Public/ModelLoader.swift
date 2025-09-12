import Foundation
import MLXLMCommon
import MLXLLM
import Hub

/// Model loader that handles downloading and loading models with progress reporting.
public final class ModelLoader {
    
    private let hubApi: HubApi
    private var modelCache: [String: ModelContainer] = [:]
    private let cacheQueue = DispatchQueue(label: "com.openFoundationModels.modelLoader.cache")
    
    public init(hubApi: HubApi = HubApi()) {
        self.hubApi = hubApi
    }
    
    /// Load a model with optional progress reporting
    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        
        if let cached = getCachedModel(modelID) {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 100
            progress?.localizedDescription = NSLocalizedString("Model loaded from cache", comment: "")
            return cached
        }
        
        let loadingProgress = progress ?? Progress(totalUnitCount: 100)
        loadingProgress.localizedDescription = NSLocalizedString("Loading model...", comment: "")
        
        let config = ModelConfiguration(id: modelID)
        
        let container = try await MLXLMCommon.loadModelContainer(
            hub: hubApi,
            configuration: config
        ) { hubProgress in
            let fraction = hubProgress.fractionCompleted
            loadingProgress.completedUnitCount = Int64(fraction * Double(loadingProgress.totalUnitCount))
            
            if let description = hubProgress.localizedAdditionalDescription {
                loadingProgress.localizedAdditionalDescription = description
            }
        }
        
        setCachedModel(container, for: modelID)
        
        loadingProgress.completedUnitCount = loadingProgress.totalUnitCount
        loadingProgress.localizedDescription = NSLocalizedString("Model ready", comment: "")
        
        return container
    }
    
    /// Download a model without loading it into memory
    public func downloadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws {
        _ = try await loadModel(modelID, progress: progress)
    }
    
    /// Load a model from a local directory
    public func loadLocalModel(from path: URL) async throws -> ModelContainer {
        let config = ModelConfiguration(directory: path)
        
        let container = try await MLXLMCommon.loadModelContainer(
            hub: hubApi,
            configuration: config
        ) { _ in
        }
        
        return container
    }
    
    public func cachedModels() -> [String] {
        cacheQueue.sync {
            Array(modelCache.keys)
        }
    }
    
    public func clearCache() {
        cacheQueue.sync {
            modelCache.removeAll()
        }
    }
    
    public func clearCache(for modelID: String) {
        _ = cacheQueue.sync {
            modelCache.removeValue(forKey: modelID)
        }
    }
    
    public func isCached(_ modelID: String) -> Bool {
        cacheQueue.sync {
            modelCache[modelID] != nil
        }
    }
    
    
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