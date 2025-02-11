import os
import json
import numpy as np
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense

# -----------------------
# 全局配置，自定义检测和详细分类的模型，以及数据集大小
# -----------------------
CONFIG = {
    "data_file": "ts_dataset.json",
    "unknown_data_file": "unknown_dataset.json",
    "report_file": "report.md",
    "test_size": 0.2,
    "seed": 42,
    "confidence_threshold": 0.8,  # 若预测的最高概率低于此值，则认为样本不属于已知类别
    "unknown_margin": 0.05,  # 若最高概率与第二高的差值小于此值，则也归为 unknown，即模型不清楚具体是哪个类别
    "detection_temperature": 3.0,  # 平坦化概率分布参数
    "num_samples": 3000,  # 需要生产的数据集样本数
    "num_classes": 8,  # 已知类别数为8（000～111）
    # 随机生成长度为3~5的时间序列
    "min_seq_length": 3,
    "max_seq_length": 5,
    # 用于检测未知类别的模型列表：
    "detection_models": ["RandomForest", "SVM", "LogisticRegression", "GradientBoosting", "KNN", "LSTM"],
    # 用于详细分类的模型列表
    "classification_models": ["SVM", "LogisticRegression", "GradientBoosting", "KNN", "LSTM"],
    "num_unknown_samples": 300  # 生成未知样本数量
}


# -----------------------
# 数据生成及管理
# -----------------------
class TimeSeriesGenerator:
    def __init__(self):
        self.num_samples = CONFIG["num_samples"]
        self.num_classes = CONFIG["num_classes"]

    def _generate_timestamp(self, base_hour, step):
        total_sec = base_hour * 3600 + step
        hours = total_sec // 3600
        mins = (total_sec % 3600) // 60
        secs = total_sec % 60
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    def _to_binary_list(self, label):
        bin_str = format(label, '03b')
        return [int(ch) for ch in bin_str]

    def _generate_event_features(self, label, seq_length):
        # 利用正弦函数生成特征，使数据具有随机性和一定模式
        features = []
        base_hour = random.randint(0, 23)
        amps = [random.uniform(0.5, 1.5) for _ in range(4)]
        phases = [random.uniform(0, 2 * np.pi) for _ in range(4)]
        offset = label * 0.5
        sigma = 0.5
        for t in range(seq_length):
            ts = self._generate_timestamp(base_hour, t)
            row = [ts]
            for i in range(4):
                val = amps[i] * np.sin(2 * np.pi * t / seq_length + phases[i]) + offset + random.gauss(0, sigma)
                row.append(round(val, 3))
            features.append(row)
        return features

    def generate_event(self, label):
        seq_length = random.randint(CONFIG["min_seq_length"], CONFIG["max_seq_length"])
        return {
            "features": self._generate_event_features(label, seq_length),
            "labels": self._to_binary_list(label)
        }

    def generate_unknown_event(self):
        seq_length = random.randint(CONFIG["min_seq_length"], CONFIG["max_seq_length"])
        features = []
        base_hour = random.randint(0, 23)
        for t in range(seq_length):
            ts = self._generate_timestamp(base_hour, t)
            # 修改未知样本的数值生成范围为 [8,18]
            row = [ts] + [round(random.uniform(8, 18), 3) for _ in range(4)]
            features.append(row)
        return {
            "features": features,
            "labels": "unknown"
        }

    def validate_event(self, event):
        for idx, row in enumerate(event["features"]):
            if len(row) != 5:
                raise ValueError(f"事件中第 {idx} 行特征长度错误：期望5，实际{len(row)}")
            if not isinstance(row[0], str):
                raise TypeError(f"事件中第 {idx} 行时间格式错误")
            if not all(isinstance(x, (int, float)) for x in row[1:]):
                raise TypeError(f"事件中第 {idx} 行数值特征类型错误")
        if isinstance(event["labels"], list):
            if len(event["labels"]) != 3 or not all(x in [0, 1] for x in event["labels"]):
                raise ValueError("事件标签格式错误，要求为长度为3的二进制列表")


class DataManager:
    def __init__(self):
        self.generator = TimeSeriesGenerator()

    def generate_full_dataset(self):
        num_events = CONFIG["num_samples"]
        num_classes = CONFIG["num_classes"]
        base_count = num_events // num_classes
        remainder = num_events % num_classes
        labels_list = []
        for cls in range(num_classes):
            count = base_count + (1 if cls < remainder else 0)
            labels_list.extend([cls] * count)
        random.seed(CONFIG["seed"])
        random.shuffle(labels_list)
        dataset = {}
        for i, cls in enumerate(labels_list):
            event = self.generator.generate_event(cls)
            self.generator.validate_event(event)
            dataset[f"event_{i + 1}"] = event
        return dataset

    def generate_unknown_dataset(self, num_unknown=None):
        if num_unknown is None:
            num_unknown = CONFIG.get("num_unknown_samples", 10)
        dataset = {}
        for i in range(num_unknown):
            event = self.generator.generate_unknown_event()
            dataset[f"unknown_event_{i + 1}"] = event
        return dataset

    def save_dataset(self, data):
        with open(CONFIG["data_file"], 'w') as f:
            json.dump(data, f, indent=2)

    def save_unknown_dataset(self, data):
        with open(CONFIG["unknown_data_file"], 'w') as f:
            json.dump(data, f, indent=2)

    def load_dataset(self):
        with open(CONFIG["data_file"], 'r') as f:
            data = json.load(f)
            for event in data.values():
                self.generator.validate_event(event)
            return data

    def load_unknown_dataset(self):
        with open(CONFIG["unknown_data_file"], 'r') as f:
            data = json.load(f)
            return data

    def dataset_exists(self):
        return os.path.exists(CONFIG["data_file"])

    def unknown_dataset_exists(self):
        return os.path.exists(CONFIG["unknown_data_file"])


# -----------------------
# 特征提取器
# -----------------------
class FeatureExtractor:
    @staticmethod
    def _time_to_fraction(time_str):
        h, m, s = map(int, time_str.split(':'))
        seconds = h * 3600 + m * 60 + s
        return seconds / 86400.0

    @staticmethod
    def extract_features(raw_data):
        X, y = [], []
        for event in raw_data.values():
            seq = []
            for row in event["features"]:
                time_val = FeatureExtractor._time_to_fraction(row[0])
                numeric_vals = row[1:]
                seq.append([time_val] + numeric_vals)
            seq = np.array(seq)
            feat_vector = []
            for col in range(5):
                column = seq[:, col]
                mean_val = np.mean(column)
                std_val = np.std(column)
                rng_val = np.max(column) - np.min(column)
                median_val = np.median(column)
                slope = np.polyfit(range(len(column)), column, 1)[0] if len(column) > 1 else 0.0
                feat_vector.extend([mean_val, std_val, rng_val, median_val, slope])
            X.append(feat_vector)
            if isinstance(event["labels"], list):
                label_int = int("".join(str(x) for x in event["labels"]), 2)
            else:
                label_int = "unknown"
            y.append(label_int)
        return np.array(X), np.array(y)

    @staticmethod
    def extract_sequence_data(raw_data):
        X_seq, y = [], []
        for event in raw_data.values():
            seq = []
            for row in event["features"]:
                time_val = FeatureExtractor._time_to_fraction(row[0])
                numeric_vals = row[1:]
                seq.append([time_val] + numeric_vals)
            X_seq.append(seq)
            if isinstance(event["labels"], list):
                label_int = int("".join(str(x) for x in event["labels"]), 2)
            else:
                label_int = "unknown"
            y.append(label_int)
        return X_seq, np.array(y)


# -----------------------
# 机器学习实验：支持检测与详细分类
# -----------------------
class MLExperiment:
    def __init__(self):
        model_mapping = {
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(probability=True, max_iter=1000),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "GradientBoosting": GradientBoostingClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        # 用于检测未知类别的模型：这里传统模型和 LSTM均参与检测
        detection_names = CONFIG.get("detection_models", [])
        self.detection_models = {}
        for name in detection_names:
            if name == "LSTM":
                self.detection_models["LSTM"] = None  # 待训练后赋值
            elif name in model_mapping:
                self.detection_models[name] = model_mapping[name]
        # 用于详细分类的模型（传统模型部分，不包含 LSTM）
        cls_names = [name for name in CONFIG.get("classification_models", []) if
                     name != "LSTM" and name in model_mapping]
        self.classification_models = {name: model_mapping[name] for name in cls_names}
        self.use_lstm = "LSTM" in CONFIG.get("classification_models", [])
        self.scaler = StandardScaler()
        self.fitted_classification_models = {}
        self.fitted_lstm_model = None  # 用于详细分类和未知检测（LSTM部分）

    def run_experiment(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=y
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        overall_results = {}
        per_model_class_metrics = {}
        unique_classes = np.unique(y_test)
        for name, model in self.classification_models.items():
            model.fit(X_train_scaled, y_train)
            self.fitted_classification_models[name] = model
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            overall_results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "confidence": np.max(y_proba, axis=1).mean()
            }
            per_class = {}
            p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(y_test, y_pred, labels=unique_classes,
                                                                     zero_division=0)
            for i, cls in enumerate(unique_classes):
                idx = (y_test == cls)
                acc_cls = np.sum(y_pred[idx] == cls) / np.sum(idx) if np.sum(idx) > 0 else 0.0
                avg_conf = np.mean(np.max(y_proba[idx], axis=1)) if np.sum(idx) > 0 else 0.0
                per_class[cls] = {
                    "accuracy": acc_cls,
                    "precision": p_arr[i],
                    "recall": r_arr[i],
                    "f1": f_arr[i],
                    "confidence": avg_conf
                }
            per_model_class_metrics[name] = per_class
        return overall_results, per_model_class_metrics

    def run_lstm_experiment(self, X_seq, y):
        X_train_seq, X_test_seq, y_train, y_test = train_test_split(
            X_seq, y, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=y
        )
        X_train_pad = pad_sequences(X_train_seq, dtype='float32', padding='post')
        X_test_pad = pad_sequences(X_test_seq, dtype='float32', padding='post')
        input_shape = X_train_pad.shape[1:]
        num_classes = CONFIG["num_classes"]
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=input_shape))
        model.add(LSTM(64, activation='tanh'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train_pad, y_train, epochs=30, batch_size=8,
                            validation_split=0.2, verbose=0)
        test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
        y_pred_prob = model.predict(X_test_pad)
        y_pred = np.argmax(y_pred_prob, axis=1)
        lstm_result = {
            "accuracy": test_acc,
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confidence": np.max(y_pred_prob, axis=1).mean()
        }
        per_class = {}
        unique_classes = np.unique(y_test)
        p_arr, r_arr, f_arr, _ = precision_recall_fscore_support(y_test, y_pred, labels=unique_classes, zero_division=0)
        for i, cls in enumerate(unique_classes):
            idx = (y_test == cls)
            acc_cls = np.sum(y_pred[idx] == cls) / np.sum(idx) if np.sum(idx) > 0 else 0.0
            avg_conf = np.mean(np.max(y_pred_prob[idx], axis=1)) if np.sum(idx) > 0 else 0.0
            per_class[cls] = {
                "accuracy": acc_cls,
                "precision": p_arr[i],
                "recall": r_arr[i],
                "f1": f_arr[i],
                "confidence": avg_conf
            }
        self.fitted_lstm_model = model
        self.detection_models["LSTM"] = model
        return lstm_result, history, per_class, model

    def _calibrate_probs(self, probs, T):
        p_cal = np.power(probs, 1.0 / T)
        p_cal = p_cal / np.sum(p_cal, axis=1, keepdims=True)
        return p_cal

    def evaluate_unknown_traditional(self, X_unknown):
        X_unknown_scaled = self.scaler.transform(X_unknown)
        unknown_results = {}
        for name, model in self.fitted_classification_models.items():
            prob = model.predict_proba(X_unknown_scaled)
            p_cal = self._calibrate_probs(prob, CONFIG["detection_temperature"])
            predictions = []
            for p in p_cal:
                sorted_p = np.sort(p)
                max_val = sorted_p[-1]
                second_val = sorted_p[-2] if len(sorted_p) >= 2 else 0.0
                if max_val < CONFIG["confidence_threshold"] or (max_val - second_val) < CONFIG["unknown_margin"]:
                    predictions.append("unknown")
                else:
                    predictions.append(np.argmax(p))
            detection_rate = np.mean([1 if pred == "unknown" else 0 for pred in predictions])
            unknown_results[name] = detection_rate
        return unknown_results

    def evaluate_unknown_lstm(self, X_unknown_seq, lstm_model):
        X_unknown_pad = pad_sequences(X_unknown_seq, dtype='float32', padding='post')
        prob = lstm_model.predict(X_unknown_pad)
        p_cal = self._calibrate_probs(prob, CONFIG["detection_temperature"])
        predictions = []
        for p in p_cal:
            sorted_p = np.sort(p)
            max_val = sorted_p[-1]
            second_val = sorted_p[-2] if len(sorted_p) >= 2 else 0.0
            if max_val < CONFIG["confidence_threshold"] or (max_val - second_val) < CONFIG["unknown_margin"]:
                predictions.append("unknown")
            else:
                predictions.append(np.argmax(p))
        detection_rate = np.mean([1 if pred == "unknown" else 0 for pred in predictions])
        return detection_rate


# -----------------------
# 报告生成器（均为静态方法，不使用 self）
# -----------------------
class ReportGenerator:
    @staticmethod
    def generate_report(classical_results, lstm_result, dataset_info, overall_per_class, unknown_traditional,
                        unknown_lstm):
        lines = []
        lines.append("# 时间序列分类实验报告")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        total_samples = dataset_info.get("total_samples",
                                         dataset_info.get("train_count", 0) + dataset_info.get("test_count", 0))
        lines.append(f"**数据集总样本数**: {total_samples}")
        lines.append(f"**测试集比例**: {CONFIG['test_size'] * 100}%")

        # 模型配置说明
        detection_models = CONFIG.get("detection_models", [])
        classification_models = CONFIG.get("classification_models", [])
        lines.append("\n## 模型配置")
        lines.append(f"- 用于检测类别特征的模型：{'、'.join(detection_models)}")
        lines.append(f"- 用于详细分类的模型：{'、'.join(classification_models)}")

        # 数据集构成
        lines.append("\n## 数据集构成")
        headers = ["Label", "样本数"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        samples_per_class = dataset_info.get("samples_per_class", {})
        for cls in sorted(samples_per_class.keys(), key=lambda x: int(x)):
            binary_label = format(cls, '03b')
            lines.append(f"| {binary_label} | {samples_per_class[cls]} |")
        lines.append(
            f"\n训练集样本数: {dataset_info.get('train_count', '未知')}, 测试集样本数: {dataset_info.get('test_count', '未知')}")

        # 总体性能表：以 Accuracy 最大值判断最佳模型
        all_results = classical_results.copy()
        all_results["LSTM"] = lstm_result
        best_accuracy = max(metrics['accuracy'] for metrics in all_results.values())
        lines.append("\n## 各模型总体性能对比")
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Avg Confidence"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for name, metrics in all_results.items():
            if round(metrics['accuracy'], 6) == round(best_accuracy, 6):
                row = [f"<span style='color:red; font-weight:bold;'>{name}</span>",
                       f"<span style='color:red; font-weight:bold;'>{metrics['accuracy'] * 100:.2f}%</span>",
                       f"<span style='color:red; font-weight:bold;'>{metrics['precision'] * 100:.2f}%</span>",
                       f"<span style='color:red; font-weight:bold;'>{metrics['recall'] * 100:.2f}%</span>",
                       f"<span style='color:red; font-weight:bold;'>{metrics['f1'] * 100:.2f}%</span>",
                       f"<span style='color:red; font-weight:bold;'>{metrics['confidence'] * 100:.2f}%</span>"]
            else:
                row = [name,
                       f"{metrics['accuracy'] * 100:.2f}%",
                       f"{metrics['precision'] * 100:.2f}%",
                       f"{metrics['recall'] * 100:.2f}%",
                       f"{metrics['f1'] * 100:.2f}%",
                       f"{metrics['confidence'] * 100:.2f}%"]
            row_str = "| " + " | ".join(row) + " |"
            lines.append(row_str)

        # 备注：Avg Confidence 的计算说明
        lines.append(
            "\n**注：Avg Confidence 为测试集中每个样本预测结果中，概率最高的值（通过 predict_proba 获得）的平均值。**\n")
        lines.append(
            "> 具体计算方式：对于每个测试样本，模型调用 predict_proba 得到各类别的概率向量；取该向量中最大概率作为该样本的置信度；所有样本的这些最大概率的平均值即为 Avg Confidence。\n")

        # 各 Label 性能对比
        lines.append("## 测试集各 Label 性能对比")
        for cls in sorted(overall_per_class.keys(), key=lambda x: int(x)):
            binary_label = format(cls, '03b')
            lines.append(f"\n### Label {binary_label}")
            headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Avg Confidence"]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            model_metrics = overall_per_class[cls]
            best_label_accuracy = max(metrics['accuracy'] for metrics in model_metrics.values())
            for model, metrics in model_metrics.items():
                if round(metrics['accuracy'], 6) == round(best_label_accuracy, 6):
                    row = [f"<span style='color:red; font-weight:bold;'>{model}</span>",
                           f"<span style='color:red; font-weight:bold;'>{metrics['accuracy'] * 100:.2f}%</span>",
                           f"<span style='color:red; font-weight:bold;'>{metrics['precision'] * 100:.2f}%</span>",
                           f"<span style='color:red; font-weight:bold;'>{metrics['recall'] * 100:.2f}%</span>",
                           f"<span style='color:red; font-weight:bold;'>{metrics['f1'] * 100:.2f}%</span>",
                           f"<span style='color:red; font-weight:bold;'>{metrics['confidence'] * 100:.2f}%</span>"]
                else:
                    row = [model,
                           f"{metrics['accuracy'] * 100:.2f}%",
                           f"{metrics['precision'] * 100:.2f}%",
                           f"{metrics['recall'] * 100:.2f}%",
                           f"{metrics['f1'] * 100:.2f}%",
                           f"{metrics['confidence'] * 100:.2f}%"]
                row_str = "| " + " | ".join(row) + " |"
                lines.append(row_str)

        # 未知类别检测结果：采用所有检测模型（经过温度校准）的检测率
        lines.append("\n## 未知类别检测")
        headers = ["Model", "Unknown Detection Rate"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        unknown_results = unknown_traditional.copy()
        unknown_results["LSTM"] = unknown_lstm
        best_unknown_rate = max(unknown_results.values()) if unknown_results else 0
        for model, rate in unknown_results.items():
            if round(rate, 6) == round(best_unknown_rate, 6):
                row = [f"<span style='color:red; font-weight:bold;'>{model}</span>",
                       f"<span style='color:red; font-weight:bold;'>{rate * 100:.2f}%</span>"]
            else:
                row = [model, f"{rate * 100:.2f}%"]
            row_str = "| " + " | ".join(row) + " |"
            lines.append(row_str)

        # 训练和测试说明
        lines.append("\n## 训练和测试说明")
        lines.append("本实验中，训练阶段仅使用已知类别（000至111）的样本进行训练；")
        lines.append("测试阶段，系统首先判断样本是否具有明显的类别特征，对应于有类别特征的样本，系统调用各模型的 predict_proba 得到各类别的"
                     "概率向量，并比较预测结果中最高概率与第二高概率的差值以及最高概率是否高于预设阈值（Confidence Threshold），若最高概率低"
                     "于该阈值，或者最高概率与第二高概率之差小于设定的 Margin，则认为该样本不属于已知类别，从而归为未知类别，否则，系统将根据模"
                     "型的预测结果对样本进行具体的分类判别。")
        lines.append("其中，用于检测类别特征的模型包括： " + "、".join(detection_models) + "；")
        lines.append("用于详细分类的模型包括： " + "、".join(classification_models) + "（部分模型可同时用于检测和分类）。")

        return "\n".join(lines)


# -----------------------
# 主函数入口
# -----------------------
def main():
    print("=== 不定长时间序列分类系统 ===")

    data_manager = DataManager()
    feature_extractor = FeatureExtractor()
    experiment = MLExperiment()

    try:
        if data_manager.dataset_exists():
            print("加载现有数据集...")
            raw_data = data_manager.load_dataset()
        else:
            print("生成新数据集...")
            raw_data = data_manager.generate_full_dataset()
            data_manager.save_dataset(raw_data)
            print(f"数据集已保存至 {CONFIG['data_file']}")

        if data_manager.unknown_dataset_exists():
            print("加载未知数据集...")
            unknown_data = data_manager.load_unknown_dataset()
        else:
            print("生成未知数据集...")
            unknown_data = data_manager.generate_unknown_dataset()
            data_manager.save_unknown_dataset(unknown_data)
            print(f"未知数据集已保存至 {CONFIG['unknown_data_file']}")

        # 提取已知数据特征
        X_fixed, y_fixed = FeatureExtractor.extract_features(raw_data)
        X_seq, y_seq = FeatureExtractor.extract_sequence_data(raw_data)

        # 计算数据集构成信息
        samples_per_class = {}
        for label in y_fixed:
            samples_per_class[label] = samples_per_class.get(label, 0) + 1
        total_samples = len(y_fixed)
        _, X_test_dummy, _, y_test_dummy = train_test_split(
            X_fixed, y_fixed, test_size=CONFIG["test_size"], random_state=CONFIG["seed"], stratify=y_fixed
        )
        dataset_info = {
            "num_classes": CONFIG["num_classes"],
            "samples_per_class": samples_per_class,
            "train_count": len(y_fixed) - len(y_test_dummy),
            "test_count": len(y_test_dummy),
            "total_samples": total_samples
        }

        print("\n开始传统模型训练...")
        classical_results, classical_per_model = experiment.run_experiment(X_fixed, y_fixed)

        print("\n开始 LSTM 模型训练...")
        lstm_result, lstm_history, lstm_per_model, lstm_model = experiment.run_lstm_experiment(X_seq, y_seq)

        overall_per_class = {}
        unique_classes = np.unique(y_fixed)
        for cls in unique_classes:
            overall_per_class[cls] = {}
        for model, cls_dict in classical_per_model.items():
            for cls, metrics in cls_dict.items():
                overall_per_class[cls][model] = metrics
        for cls, metrics in lstm_per_model.items():
            overall_per_class[cls]["LSTM"] = metrics

        # 未知样本评估：同时计算传统模型（经过温度校准）和 LSTM 的未知检测率
        X_unknown_fixed, _ = FeatureExtractor.extract_features(unknown_data)
        X_unknown_seq, _ = FeatureExtractor.extract_sequence_data(unknown_data)
        unknown_traditional = experiment.evaluate_unknown_traditional(X_unknown_fixed)
        unknown_lstm = experiment.evaluate_unknown_lstm(X_unknown_seq, lstm_model)

        report = ReportGenerator.generate_report(classical_results, lstm_result, dataset_info, overall_per_class,
                                                 unknown_traditional, unknown_lstm)
        with open(CONFIG["report_file"], 'w', encoding="utf-8") as f:
            f.write(report)
        print(f"\n实验报告已生成：{CONFIG['report_file']}")

    except Exception as e:
        print(f"\n错误发生：{str(e)}")
        print("建议删除数据集后重新生成。")


if __name__ == "__main__":
    main()
