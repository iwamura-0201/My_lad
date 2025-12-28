"""
GPU対応LSH（Locality Sensitive Hashing）モジュール

CuPyを使用してGPU上でMinHashベースのLSHを高速に計算する。
CuPyがインストールされていない場合はCPUにフォールバックする。
"""

import hashlib
from typing import List, Optional
import numpy as np

# CuPyのインポートを試行（なければCPUフォールバック）
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    print("CuPy not available. GPU LSH will fall back to CPU.")


# LSH用のパラメータ
LSH_HASH_SIZE = 8
LSH_NUM_PERMUTATIONS = 128
LSH_SHINGLE_SIZE = 3
LSH_NUM_BANDS = 16


def _generate_shingles(text: str, shingle_size: int = LSH_SHINGLE_SIZE) -> List[str]:
    """文字列からshingle（n-gram）を生成"""
    if not text or len(text) < shingle_size:
        return [text] if text else []
    return [text[i:i + shingle_size] for i in range(len(text) - shingle_size + 1)]


def _hash_shingle(shingle: str, seed: int) -> int:
    """shingleとシードを組み合わせてハッシュ値を計算"""
    combined = f"{shingle}_{seed}"
    # 64ビット整数に収める
    return int(hashlib.md5(combined.encode('utf-8')).hexdigest()[:16], 16)


def _compute_single_lsh_hash(args):
    """
    単一テキストのLSHハッシュを計算（マルチプロセス用ワーカー関数）
    """
    text, hash_size, num_permutations, num_bands = args
    rows_per_band = num_permutations // num_bands
    
    if not text:
        return ''
    
    # shingleを生成
    shingles = _generate_shingles(str(text))
    if not shingles:
        return ''
    
    # MinHash署名を計算
    signature = []
    for i in range(num_permutations):
        min_hash = min(_hash_shingle(s, i) for s in shingles)
        signature.append(min_hash)
    
    # バンドハッシュを計算
    band_hashes = []
    for b in range(num_bands):
        band_start = b * rows_per_band
        band_end = band_start + rows_per_band
        band_values = signature[band_start:band_end]
        band_str = '|'.join(str(v) for v in band_values)
        band_hash = hashlib.md5(band_str.encode('utf-8')).hexdigest()[:2]
        band_hashes.append(band_hash)
    
    combined_hash = ''.join(band_hashes)[:hash_size]
    return combined_hash


def compute_lsh_hash_cpu_batch(texts: List[str], 
                                hash_size: int = LSH_HASH_SIZE,
                                num_permutations: int = LSH_NUM_PERMUTATIONS,
                                num_bands: int = LSH_NUM_BANDS,
                                use_multiprocess: bool = False,
                                n_workers: Optional[int] = None) -> List[str]:
    """
    CPU上でバッチ処理によるLSHハッシュを計算
    
    Parameters
    ----------
    texts : List[str]
        ハッシュ化する文字列のリスト
    hash_size : int
        出力ハッシュのサイズ
    num_permutations : int
        MinHashの順列数
    num_bands : int
        LSHバンド数
    use_multiprocess : bool
        マルチプロセス並列化を使用するかどうか（デフォルト: False）
    n_workers : int, optional
        並列ワーカー数（デフォルト: CPUコア数）
        
    Returns
    -------
    List[str]
        LSHハッシュ値のリスト
    """
    if use_multiprocess:
        import multiprocessing as mp
        import os
        
        # ワーカー数の決定
        if n_workers is None:
            n_workers = os.cpu_count() or 4
        
        # 引数のリストを作成
        args_list = [(text, hash_size, num_permutations, num_bands) for text in texts]
        
        total = len(texts)
        print(f"  CPU LSH (multiprocess): Processing {total} texts with {n_workers} workers...")
        
        # マルチプロセスで並列処理
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_compute_single_lsh_hash, args_list)
        
        print(f"  CPU LSH (multiprocess): Complete!")
        return results
    
    # シングルプロセス版
    results = []
    rows_per_band = num_permutations // num_bands
    
    for text in texts:
        if not text:
            results.append('')
            continue
            
        # shingleを生成
        shingles = _generate_shingles(str(text))
        if not shingles:
            results.append('')
            continue
        
        # MinHash署名を計算
        signature = []
        for i in range(num_permutations):
            min_hash = min(_hash_shingle(s, i) for s in shingles)
            signature.append(min_hash)
        
        # バンドハッシュを計算
        band_hashes = []
        for b in range(num_bands):
            band_start = b * rows_per_band
            band_end = band_start + rows_per_band
            band_values = signature[band_start:band_end]
            band_str = '|'.join(str(v) for v in band_values)
            band_hash = hashlib.md5(band_str.encode('utf-8')).hexdigest()[:2]
            band_hashes.append(band_hash)
        
        combined_hash = ''.join(band_hashes)[:hash_size]
        results.append(combined_hash)
    
    return results


def compute_lsh_hash_gpu_batch(texts: List[str],
                                hash_size: int = LSH_HASH_SIZE,
                                num_permutations: int = LSH_NUM_PERMUTATIONS,
                                num_bands: int = LSH_NUM_BANDS,
                                batch_size: int = 10000,
                                use_multiprocess: bool = False,
                                n_workers: Optional[int] = None) -> List[str]:
    """
    GPU上でバッチ処理によるLSHハッシュを計算
    
    CuPyを使用してMinHashの計算をGPU上で並列化する。
    CuPyがない場合はCPU版にフォールバックする。
    
    Parameters
    ----------
    texts : List[str]
        ハッシュ化する文字列のリスト
    hash_size : int
        出力ハッシュのサイズ
    num_permutations : int
        MinHashの順列数
    num_bands : int
        LSHバンド数
    batch_size : int
        GPU処理のバッチサイズ
    use_multiprocess : bool
        CPUフォールバック時にマルチプロセスを使用するかどうか
    n_workers : int, optional
        並列ワーカー数（デフォルト: CPUコア数）
        
    Returns
    -------
    List[str]
        LSHハッシュ値のリスト
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available, falling back to CPU implementation.")
        return compute_lsh_hash_cpu_batch(texts, hash_size, num_permutations, num_bands, 
                                          use_multiprocess=use_multiprocess, n_workers=n_workers)
    
    results = []
    total = len(texts)
    rows_per_band = num_permutations // num_bands
    
    # 事前にハッシュシードを生成（GPU上で使用）
    # 各順列のシード値を事前計算
    np.random.seed(42)
    hash_a = np.random.randint(1, 2**31 - 1, size=num_permutations, dtype=np.int64)
    hash_b = np.random.randint(0, 2**31 - 1, size=num_permutations, dtype=np.int64)
    prime = 2**61 - 1  # メルセンヌ素数
    
    # GPU用の定数を転送
    hash_a_gpu = cp.asarray(hash_a)
    hash_b_gpu = cp.asarray(hash_b)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = texts[batch_start:batch_end]
        
        # バッチ内の各テキストを処理
        batch_signatures = []
        
        for text in batch_texts:
            if not text:
                batch_signatures.append(None)
                continue
            
            # shingleを生成してハッシュ値に変換（CPU上で）
            shingles = _generate_shingles(str(text))
            if not shingles:
                batch_signatures.append(None)
                continue
            
            # shingleのハッシュ値をnumpy配列に変換
            shingle_hashes = np.array([
                int(hashlib.md5(s.encode('utf-8')).hexdigest()[:16], 16) % prime
                for s in shingles
            ], dtype=np.int64)
            
            # GPU上でMinHash計算
            shingle_hashes_gpu = cp.asarray(shingle_hashes)
            
            # 全順列に対してMinHashを計算
            # hash_values[i, j] = (hash_a[i] * shingle_hashes[j] + hash_b[i]) % prime
            hash_matrix = (hash_a_gpu[:, cp.newaxis] * shingle_hashes_gpu + hash_b_gpu[:, cp.newaxis]) % prime
            signatures_gpu = cp.min(hash_matrix, axis=1)
            
            batch_signatures.append(cp.asnumpy(signatures_gpu))
        
        # バンドハッシュを計算（CPU上で最終処理）
        for signature in batch_signatures:
            if signature is None:
                results.append('')
                continue
            
            band_hashes = []
            for b in range(num_bands):
                band_start = b * rows_per_band
                band_end = band_start + rows_per_band
                band_values = signature[band_start:band_end]
                band_str = '|'.join(str(int(v)) for v in band_values)
                band_hash = hashlib.md5(band_str.encode('utf-8')).hexdigest()[:2]
                band_hashes.append(band_hash)
            
            combined_hash = ''.join(band_hashes)[:hash_size]
            results.append(combined_hash)
        
        # 進捗表示
        if batch_end % (batch_size * 10) == 0 or batch_end == total:
            print(f"  GPU LSH Progress: {batch_end}/{total} ({100*batch_end/total:.1f}%)")
    
    return results


class LSHProcessor:
    """
    LSH処理を管理するクラス
    
    GPUの可用性を自動検出し、最適な処理方法を選択する。
    """
    
    def __init__(self, 
                 hash_size: int = LSH_HASH_SIZE,
                 num_permutations: int = LSH_NUM_PERMUTATIONS,
                 num_bands: int = LSH_NUM_BANDS,
                 batch_size: int = 10000,
                 use_gpu: bool = True):
        """
        Parameters
        ----------
        hash_size : int
            出力ハッシュのサイズ
        num_permutations : int
            MinHashの順列数
        num_bands : int
            LSHバンド数
        batch_size : int
            バッチサイズ
        use_gpu : bool
            GPUを使用するかどうか（CuPyが利用可能な場合のみ有効）
        """
        self.hash_size = hash_size
        self.num_permutations = num_permutations
        self.num_bands = num_bands
        self.batch_size = batch_size
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        if self.use_gpu:
            print(f"LSHProcessor initialized with GPU support (CuPy)")
        else:
            print(f"LSHProcessor initialized with CPU only")
    
    def compute_batch(self, texts: List[str]) -> List[str]:
        """
        テキストのリストに対してLSHハッシュを計算
        
        Parameters
        ----------
        texts : List[str]
            ハッシュ化する文字列のリスト
            
        Returns
        -------
        List[str]
            LSHハッシュ値のリスト
        """
        if self.use_gpu:
            return compute_lsh_hash_gpu_batch(
                texts,
                hash_size=self.hash_size,
                num_permutations=self.num_permutations,
                num_bands=self.num_bands,
                batch_size=self.batch_size
            )
        else:
            return compute_lsh_hash_cpu_batch(
                texts,
                hash_size=self.hash_size,
                num_permutations=self.num_permutations,
                num_bands=self.num_bands
            )
    
    def compute_single(self, text: str) -> str:
        """
        単一のテキストに対してLSHハッシュを計算
        
        Parameters
        ----------
        text : str
            ハッシュ化する文字列
            
        Returns
        -------
        str
            LSHハッシュ値
        """
        results = self.compute_batch([text])
        return results[0] if results else ''


# グローバルインスタンス（簡易使用向け）
_default_processor: Optional[LSHProcessor] = None


def get_lsh_processor(use_gpu: bool = True) -> LSHProcessor:
    """デフォルトのLSHProcessorを取得"""
    global _default_processor
    if _default_processor is None:
        _default_processor = LSHProcessor(use_gpu=use_gpu)
    return _default_processor


def compute_lsh_gpu(text: str) -> str:
    """
    GPU対応のLSHハッシュを計算（単一テキスト用の簡易関数）
    
    Parameters
    ----------
    text : str
        ハッシュ化する文字列
        
    Returns
    -------
    str
        LSHハッシュ値
    """
    processor = get_lsh_processor()
    return processor.compute_single(text)


if __name__ == '__main__':
    # テスト
    test_texts = [
        "SubjectUserName=taro|TargetUserName=hanako",
        "SubjectUserName=taro|TargetUserName=jiro",
        "SubjectUserName=admin|TargetUserName=system",
        "",
        "short"
    ]
    
    print("=== LSH GPU Test ===")
    print(f"CuPy available: {CUPY_AVAILABLE}")
    
    processor = LSHProcessor(use_gpu=True)
    results = processor.compute_batch(test_texts)
    
    for text, hash_val in zip(test_texts, results):
        print(f"  '{text[:40]}...' -> '{hash_val}'")
