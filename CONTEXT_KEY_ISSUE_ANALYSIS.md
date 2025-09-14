# コンテキストキー表示問題の分析

## 問題の概要

KeyDetectionLogitProcessor において、JSON 生成時のエントロピー表示で、現在のコンテキストに応じた制約キーが表示されない。すべてのコンテキストで同じキー配列が表示されてしまう。

## 現象

### 現在の誤った動作
```
# すべてのコンテキストで同じキーが表示される
[Step 810] Entropy: 0.00 🔑 KEY [Available: departments, employeeCount, founded, headquarters, name, type]
```

### 期待される正しい動作
```
# ルートレベル
[Step X] 🔑 KEY [Available: departments, employeeCount, founded, headquarters, name, type]

# headquarters オブジェクト内
[Step Y] 🔑 KEY [Available: city, country, postalCode, street]

# departments 配列内
[Step Z] 🔑 KEY [Available: headCount, manager, name, projects, type]

# departments[].manager 内
[Step W] 🔑 KEY [Available: email, firstName, lastName, level, yearsExperience]
```

## 根本原因の分析

### 1. トークン処理とコンテキスト更新のタイミング不整合

#### 問題点
- LLM はトークン単位で生成（1トークン = 複数文字の可能性）
- コンテキスト更新は文字単位で処理
- タイミングのズレによりコンテキスト更新が遅れる

#### 例
```json
{"headquarters":{"city":"SF"}}
```

トークン化の例：
- Token 1: `{"head`
- Token 2: `quarters`
- Token 3: `":{"` ← ここで headquarters コンテキストに入るべき
- Token 4: `city`

現在の実装では Token 3 の処理時に複数の状態遷移が発生：
1. `"` でキー終了
2. `:` でコロン期待
3. `{` でオブジェクト開始

これらが1トークン内で起きると、コンテキスト更新が追いつかない。

### 2. JSONContextTracker の更新ロジックの問題

#### 現在の実装の流れ
```swift
// KeyDetectionLogitProcessor.swift
if case .inString(.body(kind: .key, escaped: false)) = previousPhase {
    if case .inObject(.expectColon) = stateMachine.phase {
        // キーが完了
        contextTracker.keyDetected(keyName)  // ← キーを記録
    }
}

// その後...
if case .inObject(.expectKeyOrEnd) = currentPhase {
    if case .inObject(.expectValue) = previousPhase {
        contextTracker.enterObject()  // ← オブジェクトに入る
    }
}
```

#### 問題点
1. `keyDetected()` と `enterObject()` の呼び出しが離れている
2. `enterObject()` 時に、どのキーのオブジェクトなのか不明
3. 配列の場合の処理も同様に不正確

### 3. getCurrentContextKeys の呼び出しタイミング

#### 現在の実装
```swift
public func process(logits: MLXArray) -> MLXArray {
    // 前のステップの情報を表示
    if let pending = pendingLogitInfo {
        let contextKeys = getCurrentContextKeys()  // ← この時点のコンテキスト
        displayStep(pending, isKey: lastWasInKeyGeneration, contextKeys: contextKeys)
    }

    // 現在のステップの情報を保存
    pendingLogitInfo = buildLogitInfo(from: logits, step: stepCount)
}
```

#### 問題点
- `pendingLogitInfo` を表示する時のコンテキストが、生成時のコンテキストと異なる可能性
- 1ステップ遅れでコンテキストが更新される

## 設計上の問題

### 1. 責任の分離が不完全

現在の設計：
- **JSONStateMachine**: JSON のパース状態を管理
- **JSONContextTracker**: JSON の階層構造を管理
- **KeyDetectionLogitProcessor**: 両者を統合

問題：
- 2つのコンポーネントが同じ情報（JSON構造）を異なる方法で追跡
- 同期が取れていない

### 2. ステートレスな処理とステートフルな処理の混在

- `process(logits:)`: ステートレスであるべき（純粋な変換）
- `didSample(token:)`: ステートフル（状態を更新）

現在は両方で状態を管理しており、整合性が取れていない。

## 解決策

### 方針 1: コンテキスト更新の改善（短期的）

1. **pendingKey の導入**
   ```swift
   private var pendingKey: String? = nil

   // キー完了時
   if keyCompleted {
       pendingKey = keyName
   }

   // オブジェクト/配列開始時
   if char == "{" && pendingKey != nil {
       contextTracker.keyDetected(pendingKey!)
       contextTracker.enterObject()
       pendingKey = nil
   }
   ```

2. **LogitInfo にコンテキスト情報を保存**
   ```swift
   struct LogitInfo {
       // ... existing fields ...
       let contextPath: String
       let contextKeys: [String]
   }
   ```

### 方針 2: アーキテクチャの再設計（長期的）

1. **統合された状態管理**
   - JSONStateMachine と JSONContextTracker を統合
   - 単一の状態管理コンポーネント

2. **トークンベースの処理**
   - 文字単位ではなくトークン単位で状態を管理
   - トークンごとに完全な状態遷移を処理

3. **イベント駆動アーキテクチャ**
   ```swift
   enum JSONEvent {
       case keyStarted
       case keyCompleted(String)
       case objectStarted
       case objectEnded
       case arrayStarted
       case arrayEnded
   }
   ```

## テスト戦略

### 1. ユニットテスト
- 各コンテキストでの `getCurrentContextKeys()` の戻り値を検証
- トークン境界をまたぐケースのテスト

### 2. 統合テスト
- CompanyProfile の完全な JSON 生成
- 各ステップでの制約キー表示を検証

### 3. デバッグログ
```swift
[Context] Path: "" → Keys: [departments, employeeCount, ...]
[Context] Path: "headquarters" → Keys: [city, country, ...]
[Context] Path: "departments[]" → Keys: [headCount, manager, ...]
```

## 実装優先順位

1. **即座に修正**（優先度: 高）
   - LogitInfo にコンテキスト情報を含める
   - pendingKey による正確なコンテキスト更新

2. **次のイテレーション**（優先度: 中）
   - JSONContextTracker のテスト強化
   - デバッグログの改善

3. **将来的な改善**（優先度: 低）
   - アーキテクチャの再設計
   - トークンベースの状態管理

## 検証項目

- [ ] ルートレベルで正しいキーが表示される
- [ ] headquarters 内で city, country などが表示される
- [ ] departments[] 内で headCount, manager などが表示される
- [ ] departments[].manager 内で firstName, lastName などが表示される
- [ ] ネストから戻った時に親のコンテキストキーが表示される