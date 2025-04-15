import argparse
import os
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch_fidelity
#/root/anaconda3/envs/CDAE_FID
def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def download_file(url, file_name, save_dir,logger):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_size = int(response.headers['Content-Length'])
        logger.info(f"文件大小: {file_size} 字节")

        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=file_name)

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()
        logger.info(f"文件已成功下载到 {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        logger.info(f"下载失败：{e}")
        return None

def extract_images_from_npz(npz_file, output_dir,logger):
    with np.load(npz_file) as data:
        arr_0 = data['arr_0']
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i in tqdm(range(arr_0.shape[0]), desc="Extracting images"):
            img_array = arr_0[i]
            img = Image.fromarray(img_array)
            img.save(os.path.join(output_dir, f"image_{i:05d}.png"))
    
    logger.info(f"已从 .npz 文件中提取 {arr_0.shape[0]} 张图片到 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Calculate FID between reference images and generated images.")
    parser.add_argument("real_images_path", help="Path to reference images directory")
    parser.add_argument("generated_images_path", help="Path to generated images directory")
    parser.add_argument("output_path", help="Path to save results")
    parser.add_argument("--npz_url", default="https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz", help="URL to the .npz file")
    args = parser.parse_args()
    generated_folder_name = os.path.basename(os.path.normpath(args.generated_images_path))
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    if not os.path.exists(args.real_images_path) or len(os.listdir(args.real_images_path)) == 0:
        logger.info(f"参考图片文件夹 {args.real_images_path} 不存在或为空，开始下载 .npz 文件...")
        
        npz_file = download_file(args.npz_url, "VIRTUAL_imagenet256_labeled.npz", args.real_images_path,logger)
        if npz_file is not None:
            extract_images_from_npz(npz_file, args.real_images_path,logger)
        else:
            logger.info(f"无法下载 .npz 文件，请检查链接或网络环境。")
            return

    # 计算 FID
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=args.real_images_path,
        input2=args.generated_images_path,
        cuda=True,
        isc=False, 
        fid=True,
        verbose=False
    )

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)

    # 将结果保存到文本文件
    result_file = os.path.join(args.output_path, "fid_results.txt")
    with open(result_file, 'a') as f:
        f.write(f"Generated Folder: {generated_folder_name}\n")
        f.write(f"FID: {metrics_dict['frechet_inception_distance']}\n")
        f.write("\n")

if __name__ == "__main__":
    main()