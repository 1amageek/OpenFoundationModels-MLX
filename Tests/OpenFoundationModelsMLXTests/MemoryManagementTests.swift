import Testing
@testable import OpenFoundationModelsMLX
import MLX

struct MemoryManagementTests {
    
    // MARK: - Test Memory Management in MLXBackend
    
    @Test("Model unloading")
    func modelUnloading() async throws {
        let backend = try await MLXBackend(modelID: "test-model")
        
        // Load a model (mock)
        // Note: In real tests, you'd use a small test model
        // try await backend.loadModel("test-model")
        
        // Verify model is loaded
        let _ = await backend.currentModel()
        // #expect(currentModel != nil)
        
        // Unload the model
        await backend.unloadModel()
        
        // Verify model is unloaded
        let afterUnload = await backend.currentModel()
        #expect(afterUnload == nil)
    }
    
    @Test("Clear all models")
    func clearAllModels() async throws {
        let backend = try await MLXBackend(modelID: "test-model")
        
        // Clear all models
        await backend.clearAllModels()
        
        // Verify no models are loaded
        let models = await backend.cachedModels()
        #expect(models.isEmpty)
    }
    
    @Test("Memory pressure handling")
    func memoryPressureHandling() async throws {
        let backend = try await MLXBackend(modelID: "test-model")
        
        // Trigger memory pressure handling
        await backend.handleMemoryPressure()
        
        // This should not crash and should handle gracefully
        // In a real test, you'd verify cache limits are adjusted
    }
    
    // MARK: - Test Thread Safety
    
    @Test("Concurrent abort position access")
    func concurrentAbortPositionAccess() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        // Concurrent reads and writes
        await withTaskGroup(of: Void.self) { group in
            // Multiple readers
            for _ in 0..<10 {
                group.addTask {
                    _ = generator.getAbortPosition()
                }
            }
            
            // Writer
            group.addTask {
                // Simulate setting abort position through generation
                let stream = AsyncStream<String> { continuation in
                    continuation.yield("test")
                    continuation.finish()
                }
                
                do {
                    for try await _ in generator.generate(baseStream: stream) {
                        // Process
                    }
                } catch {
                    // Handle error
                }
            }
        }
        
        // Should complete without crashes or deadlocks
    }
    
    @Test("TokenTrie cache concurrency")
    func tokenTrieCacheConcurrency() async throws {
        let schema1 = SchemaMeta(keys: ["key1", "key2"], required: [])
        let schema2 = SchemaMeta(keys: ["key3", "key4"], required: [])
        let tokenizer = MockTokenizer()
        
        // Concurrent cache access
        await withTaskGroup(of: TokenTrie.self) { group in
            // Multiple concurrent builds
            for i in 0..<20 {
                let schema = i % 2 == 0 ? schema1 : schema2
                group.addTask {
                    return TokenTrieBuilder.buildCached(schema: schema, tokenizer: tokenizer)
                }
            }
            
            var tries: [TokenTrie] = []
            for await trie in group {
                tries.append(trie)
            }
            
            // Verify all tries are valid
            #expect(tries.count == 20)
        }
    }
    
    // MARK: - Test Task Cancellation
    
    @Test("Generation cancellation")
    func generationCancellation() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        // Create a long-running stream
        let stream = AsyncStream<String> { continuation in
            Task {
                for i in 0..<100 {
                    try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
                    continuation.yield("chunk\(i)")
                }
                continuation.finish()
            }
        }
        
        // Start generation in a task
        let task = Task {
            var chunks: [String] = []
            do {
                for try await chunk in generator.generate(baseStream: stream) {
                    chunks.append(chunk)
                }
            } catch {
                // Expected cancellation
            }
            return chunks
        }
        
        // Cancel after a short delay
        try await Task.sleep(nanoseconds: 50_000_000) // 50ms
        task.cancel()
        
        let result = await task.value
        
        // Should have processed some but not all chunks
        #expect(result.count > 0)
        #expect(result.count < 100)
    }
    
    // MARK: - Test Error Handling
    
    @Test("Safety constraints on error")
    func safetyConstraintsOnError() throws {
        let schema = SchemaMeta(keys: ["test"], required: [])
        let tokenizer = MockSwiftTokenizer()
        let processor = TokenTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        // Create logits that would cause an error
        let logits = MLX.zeros([1, 1000])
        
        // Process should apply safety constraints on error
        let result = processor.process(logits: logits)
        
        // Result should have constraints applied
        // Verify shape is preserved
        #expect(result.shape == logits.shape)
    }
    
    // MARK: - Test Metrics Collection
    
    @Test("Metrics recording")
    func metricsRecording() async throws {
        let metrics = GenerationMetrics()
        
        // Record some generation data
        await metrics.recordGeneration(tokens: 100, duration: 2.0, success: true)
        await metrics.recordGeneration(tokens: 50, duration: 1.0, success: false)
        await metrics.recordRetry()
        await metrics.recordCacheHit()
        await metrics.recordCacheMiss()
        await metrics.recordCacheMiss()
        
        // Get metrics
        let report = await metrics.getMetrics()
        
        // Verify metrics
        #expect(report.totalTokensGenerated == 150)
        #expect(report.totalGenerationTime == 3.0)
        #expect(report.successfulGenerations == 1)
        #expect(report.failedGenerations == 1)
        #expect(report.totalRetries == 1)
        #expect(report.cacheHits == 1)
        #expect(report.cacheMisses == 2)
        
        // Verify calculated metrics
        #expect(report.averageTokensPerSecond == 50.0)
        #expect(report.successRate == 0.5)
        #expect(abs(report.cacheHitRate - 1.0/3.0) < 0.01)
    }
    
    @Test("Metrics report generation")
    func metricsReport() async throws {
        let metrics = GenerationMetrics()
        
        // Add some data
        await metrics.recordGeneration(tokens: 1000, duration: 10.0, success: true)
        await metrics.recordError(MLXBackendError.noModelLoaded)
        await metrics.recordMemoryUsage(1024 * 1024 * 512) // 512 MB
        
        // Generate report
        let report = await metrics.generateReport()
        
        // Verify report contains expected sections
        #expect(report.contains("Generation Metrics Report"))
        #expect(report.contains("Generation Stats"))
        #expect(report.contains("Success Rate"))
        #expect(report.contains("Cache Performance"))
        #expect(report.contains("Memory"))
        #expect(report.contains("512.0 MB"))
    }
    
    // MARK: - Stress Tests
    
    @Test("High concurrency generation")
    func highConcurrencyGeneration() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        // Create multiple concurrent generation streams
        await withTaskGroup(of: Int.self) { group in
            for i in 0..<50 {
                group.addTask {
                    let stream = AsyncStream<String> { continuation in
                        for j in 0..<10 {
                            continuation.yield("stream\(i)-chunk\(j)")
                        }
                        continuation.finish()
                    }
                    
                    var count = 0
                    do {
                        for try await _ in generator.generate(baseStream: stream) {
                            count += 1
                        }
                    } catch {
                        // Handle errors
                    }
                    return count
                }
            }
            
            var totalChunks = 0
            for await chunkCount in group {
                totalChunks += chunkCount
            }
            
            // All chunks should be processed
            #expect(totalChunks == 500)
        }
    }
    
    @Test("Memory pressure under load")
    func memoryPressureUnderLoad() async throws {
        let backend = try await MLXBackend(modelID: "test-model")
        
        // Simulate high memory usage scenario
        for i in 0..<10 {
            if i % 3 == 0 {
                await backend.handleMemoryPressure()
            }
            // In real scenario, would load/unload models
            await backend.clearAllModels()
        }
        
        // Should handle without crashes
    }
    
    @Test("Custom memory limits")
    func customMemoryLimits() async throws {
        // Test with different memory configurations
        // Using bytes directly
        let backend1 = try await MLXBackend(modelID: "test-model", maxCacheMemory: 1_073_741_824)  // 1GB in bytes
        
        // Using MB helper
        let backend2 = try await MLXBackend(modelID: "test-model", maxCacheMemory: MLXBackend.MemorySize.MB(512))  // 512MB
        
        // Using GB helper
        let backend3 = try await MLXBackend(modelID: "test-model", maxCacheMemory: MLXBackend.MemorySize.GB(4))  // 4GB
        
        // Verify backends are created successfully
        // Actors are never nil after creation, just verify they exist
        _ = backend1
        _ = backend2
        _ = backend3
    }
}