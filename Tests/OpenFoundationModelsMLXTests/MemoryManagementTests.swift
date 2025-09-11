import Testing
@testable import OpenFoundationModelsMLX
import Foundation
import MLX
import MLXLMCommon

struct MemoryManagementTests {
    
    @Test("ModelLoader caching")
    func modelLoaderCaching() async throws {
        let loader = ModelLoader()
        
        #expect(loader.cachedModels().isEmpty)
        
        loader.clearCache()
        #expect(loader.cachedModels().isEmpty)
        
        #expect(loader.isCached("test-model") == false)
    }
    
    @Test("MLXBackend model management")
    func backendModelManagement() async throws {
        let backend = MLXBackend()
        
        let currentModel = await backend.currentModel()
        #expect(currentModel == nil)
        #expect(await backend.hasModel() == false)
        
        await backend.clearModel()
        
        #expect(await backend.currentModel() == nil)
    }
    
    @Test("Progress reporting")
    func progressReporting() async throws {
        let loader = ModelLoader()
        let progress = Progress(totalUnitCount: 100)
        
        #expect(progress.completedUnitCount == 0)
        #expect(progress.fractionCompleted == 0.0)
    }
    
    @Test("Concurrent abort position access")
    func concurrentAbortPositionAccess() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask {
                    _ = generator.getAbortPosition()
                }
            }
            
            group.addTask {
                let stream = AsyncStream<String> { continuation in
                    continuation.yield("test")
                    continuation.finish()
                }
                
                do {
                    for try await _ in generator.generate(baseStream: stream) {
                        
                    }
                } catch {
                    
                }
            }
        }
    }
    
    @Test("TokenTrie cache concurrency")
    func tokenTrieCacheConcurrency() async throws {
        let keys1 = ["key1", "key2"]
        let keys2 = ["key3", "key4"]
        let tokenizer = MockTokenizer()
        
        await withTaskGroup(of: TokenTrie.self) { group in
            for i in 0..<20 {
                let keys = i % 2 == 0 ? keys1 : keys2
                group.addTask {
                    return TokenTrieBuilder.build(keys: keys, tokenizer: tokenizer)
                }
            }
            
            var tries: [TokenTrie] = []
            for await trie in group {
                tries.append(trie)
            }
            
            #expect(tries.count == 20)
        }
    }
    
    @Test("Generation cancellation")
    func generationCancellation() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
        let stream = AsyncStream<String> { continuation in
            Task {
                for i in 0..<100 {
                    try? await Task.sleep(nanoseconds: 10_000_000)
                    continuation.yield("chunk\(i)")
                }
                continuation.finish()
            }
        }
        
        let task = Task {
            var chunks: [String] = []
            do {
                for try await chunk in generator.generate(baseStream: stream) {
                    chunks.append(chunk)
                }
            } catch {
                
            }
            return chunks
        }
        
        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        
        let result = await task.value
        
        #expect(result.count > 0)
        #expect(result.count < 100)
    }
    
    @Test("Safety constraints on error")
    func safetyConstraintsOnError() throws {
        let schema = SchemaNode(
            kind: .object,
            properties: ["test": SchemaNode.any],
            required: []
        )
        let tokenizer = MockSwiftTokenizer()
        let processor = DPDAKeyTrieLogitProcessor(schema: schema, tokenizer: tokenizer)
        
        let logits = MLX.zeros([1, 1000])
        
        let result = processor.process(logits: logits)
        
        #expect(result.shape == logits.shape)
    }
    
    @Test("Metrics recording")
    func metricsRecording() async throws {
        let metrics = GenerationMetrics()
        
        await metrics.recordGeneration(tokens: 100, duration: 2.0, success: true)
        await metrics.recordGeneration(tokens: 50, duration: 1.0, success: false)
        await metrics.recordRetry()
        await metrics.recordCacheHit()
        await metrics.recordCacheMiss()
        await metrics.recordCacheMiss()
        
        let report = await metrics.getMetrics()
        
        #expect(report.totalTokensGenerated == 150)
        #expect(report.totalGenerationTime == 3.0)
        #expect(report.successfulGenerations == 1)
        #expect(report.failedGenerations == 1)
        #expect(report.totalRetries == 1)
        #expect(report.cacheHits == 1)
        #expect(report.cacheMisses == 2)
        
        #expect(report.averageTokensPerSecond == 50.0)
        #expect(report.successRate == 0.5)
        #expect(abs(report.cacheHitRate - 1.0/3.0) < 0.01)
    }
    
    @Test("Metrics report generation")
    func metricsReport() async throws {
        let metrics = GenerationMetrics()
        
        await metrics.recordGeneration(tokens: 1000, duration: 10.0, success: true)
        await metrics.recordError(MLXBackend.MLXBackendError.noModelSet)
        await metrics.recordMemoryUsage(1024 * 1024 * 512)
        
        let report = await metrics.generateReport()
        
        #expect(report.contains("Generation Metrics Report"))
        #expect(report.contains("Generation Stats"))
        #expect(report.contains("Success Rate"))
        #expect(report.contains("Cache Performance"))
        #expect(report.contains("Memory"))
        #expect(report.contains("512.0 MB"))
    }
    
    @Test("High concurrency generation")
    func highConcurrencyGeneration() async throws {
        let processor = MockLogitProcessor()
        let generator = AbortableGenerator(processor: processor)
        
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
                        
                    }
                    return count
                }
            }
            
            var totalChunks = 0
            for await chunkCount in group {
                totalChunks += chunkCount
            }
            
            #expect(totalChunks == 500)
        }
    }
    
    @Test("Memory pressure under load")
    func memoryPressureUnderLoad() async throws {
        let backend = MLXBackend()
        
        for i in 0..<10 {
            if i % 3 == 0 {
                await backend.clearModel()
            }
        }
    }
    
    @Test("ModelLoader cache management")
    func modelLoaderCacheManagement() async throws {
        let loader = ModelLoader()
        
        loader.clearCache()
        loader.clearCache(for: "specific-model")
    }
}