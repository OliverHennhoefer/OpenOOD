import argparse
import os
import zipfile

import gdown

benchmarks_dict = {
    "bimcv": [
        "bimcv",
        "ct",
        "xraybone",
        "actmed",
        "mnist",
        "cifar10",
        "texture",
        "tin",
    ],
    "mnist": [
        "mnist",
        "notmnist",
        "fashionmnist",
        "texture",
        "cifar10",
        "tin",
        "places365",
        "cinic10",
    ],
    "cifar-10": [
        "cifar10",
        "cifar100",
        "tin",
        "mnist",
        "svhn",
        "texture",
        "places365",
        "tin597",
    ],
    "cifar-100": [
        "cifar100",
        "cifar10",
        "tin",
        "svhn",
        "texture",
        "places365",
        "tin597",
    ],
    "imagenet-200": [
        "imagenet_1k",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "texture",
        "openimage_o",
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
    ],
    "imagenet-1k": [
        "imagenet_1k",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "texture",
        "openimage_o",
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
    ],
    "misc": [
        "cifar10c",
        "fractals_and_fvis",
        "usps",
        "imagenet10",
        "hannover",
        # 'imagenet200_cae', 'imagenet200_edsr', 'imagenet200_stylized'
    ],
}

dir_dict = {
    "images_classic/": [
        "cifar100",
        "tin",
        "tin597",
        "svhn",
        "cinic10",
        "imagenet10",
        "mnist",
        "fashionmnist",
        "cifar10",
        "cifar100c",
        "places365",
        "cifar10c",
        "fractals_and_fvis",
        "usps",
        "texture",
        "notmnist",
    ],
    "images_largescale/": [
        "imagenet_1k",
        "species_sub",
        "ssb_hard",
        "ninco",
        "inaturalist",
        "places",
        "sun",
        "openimage_o",
        "imagenet_v2",
        "imagenet_c",
        "imagenet_r",
        # 'imagenet200_cae', 'imagenet200_edsr', 'imagenet200_stylized'
    ],
    "images_medical/": ["actmed", "bimcv", "ct", "hannover", "xraybone"],
}

download_id_dict = {
    "osr": "1L9MpK9QZq-o-JrFHrfo5lM4-FsFPk0e9",
    "mnist_lenet": "13mEvYF9rVIuch8u0RVDLf_JMOk3PAYCj",
    "cifar10_res18": "1rPEScK7TFjBn_W_frO-8RSPwIG6_x0fJ",
    "cifar100_res18": "1OOf88A48yXFw4fSU02XQT-3OQKf31Csy",
    "imagenet_res50": "1tgY_PsfkazLDyI1pniDMDEehntBhFyF3",
    "cifar10_res18_v1.5": "1byGeYxM_PlLjT72wZsMQvP6popJeWBgt",
    "cifar100_res18_v1.5": "1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-",
    "imagenet200_res18_v1.5": "1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs",
    "imagenet_res50_v1.5": "15PdDMNRfnJ7f2oxW6lI-Ge4QJJH3Z0Fy",
    "benchmark_imglist": "1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1",
    "usps": "1KhbWhlFlpFjEIb4wpvW0s9jmXXsHonVl",
    "cifar100": "1PGKheHUsf29leJPPGuXqzLBMwl8qMF8_",
    "cifar10": "1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1",
    "cifar10c": "170DU_ficWWmbh6O2wqELxK9jxRiGhlJH",
    "cinic10": "190gdcfbvSGbrRK6ZVlJgg5BqqED6H_nn",
    "svhn": "1DQfc11HOtB1nEwqS4pWUFp8vtQ3DczvI",
    "fashionmnist": "1nVObxjUBmVpZ6M0PPlcspsMMYHidUMfa",
    "cifar100c": "1MnETiQh9RTxJin2EHeSoIAJA28FRonHx",
    "mnist": "1CCHAGWqA1KJTFFswuF9cbhmB-j98Y1Sb",
    "fractals_and_fvis": "1EZP8RGOP-XbMsKex3r-BGI5F1WAP_PJ3",
    "tin": "1PZ-ixyx52U989IKsMA2OT-24fToTrelC",
    "tin597": "1R0d8zBcUxWNXz6CPXanobniiIfQbpKzn",
    "texture": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
    "imagenet10": "1qRKp-HCLkmfiWwR-PXthN7-2dxIQVKxP",
    "notmnist": "16ueghlyzunbksnc_ccPgEAloRW9pKO-K",
    "places365": "1Ec-LRSTf6u5vEctKX9vRp9OA6tqnJ0Ay",
    "places": "1fZ8TbPC4JGqUCm-VtvrmkYxqRNp2PoB3",
    "sun": "1ISK0STxWzWmg-_uUr4RQ8GSLFW7TZiKp",
    "species_sub": "1-JCxDx__iFMExkYRMylnGJYTPvyuX6aq",
    "imagenet_1k": "1i1ipLDFARR-JZ9argXd2-0a6DXwVhXEj",
    "ssb_hard": "1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE",
    "ninco": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "imagenet_v2": "1akg2IiE22HcbvTBpwXQoD7tgfPCdkoho",
    "imagenet_r": "1EzjMN2gq-bVV7lg-MEAdeuBuz-7jbGYU",
    "imagenet_c": "1JeXL9YH4BO8gCJ631c5BHbaSsl-lekHt",
    "imagenet_o": "1S9cFV7fGvJCcka220-pIO9JPZL1p1V8w",
    "openimage_o": "1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE",
    "inaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "actmed": "1tibxL_wt6b3BjliPaQ2qjH54Wo4ZXWYb",
    "ct": "1k5OYN4inaGgivJBJ5L8pHlopQSVnhQ36",
    "hannover": "1NmqBDlcA1dZQKOvgcILG0U1Tm6RP0s2N",
    "xraybone": "1ZzO3y1-V_IeksJXEvEfBYKRoQLLvPYe9",
    "bimcv": "1nAA45V6e0s5FAq2BJsj9QH5omoihb7MZ",
}


def require_download(filename, path):
    for item in os.listdir(path):
        if (
            item.startswith(filename)
            or filename.startswith(item)
            or path.endswith(filename)
        ):
            return False

    else:
        print(filename + " needs download:")
        return True


def download_dataset(dataset, args):
    """Downloads and extracts a specific dataset."""
    store_path_base = args.save_dir[0]  # e.g., './data'
    store_path_subdir = ""

    for key, ds_list in dir_dict.items():
        if dataset in ds_list:
            store_path_subdir = key  # e.g., 'images_classic/'
            break
    else:
        print(
            f"Warning: Dataset '{dataset}' not found in dir_dict, saving to root data dir."
        )
        # Decide on a fallback or skip? For now, save in base data dir.
        # return # Uncomment this to skip unknown datasets

    # store_path is the directory where the ZIP should be extracted TO
    store_path = os.path.join(store_path_base, store_path_subdir, dataset)
    if not os.path.exists(store_path):
        print(f"Creating directory: {store_path}")
        os.makedirs(store_path)

    # Check if the dataset *directory* already exists (meaning it was likely extracted)
    # This replaces the original require_download logic for datasets
    if os.path.exists(store_path) and any(
        os.scandir(store_path)
    ):  # Check if dir exists and is not empty
        print(
            f"Dataset directory '{store_path}' exists and is not empty, skipping download."
        )
        return

    # --- MODIFICATION 1: Construct full filepath and call gdown correctly ---
    expected_zip_name = dataset + ".zip"
    output_filepath = os.path.join(
        store_path, expected_zip_name
    )  # Full path for the zip file itself

    print(f"Attempting to download {dataset} to {output_filepath}...")
    try:
        # Pass the full filepath to gdown
        gdown.download(
            id=download_id_dict[dataset],
            output=output_filepath,
            resume=True,
            quiet=False,
        )
        # Check if download completed successfully before unzipping
        if os.path.exists(output_filepath):
            print(f"Download successful. Unzipping {output_filepath}...")
            with zipfile.ZipFile(output_filepath, "r") as zip_file:
                zip_file.extractall(store_path)  # Extract TO the containing directory
            print(f"Unzipping successful. Removing {output_filepath}...")
            os.remove(output_filepath)  # Remove the zip file after extraction
            print(
                f"Dataset '{dataset}' successfully downloaded and extracted to {store_path}."
            )
        else:
            print(
                f"ERROR: Download failed, {output_filepath} not found after gdown call."
            )
    except zipfile.BadZipFile:
        print(f"ERROR: Bad zip file downloaded: {output_filepath}. Deleting.")
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
    except Exception as e:
        print(f"ERROR during {dataset} download or processing: {e}")
        # Optionally remove potentially incomplete file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
                print(f"Removed potentially corrupt file: {output_filepath}")
            except OSError as rm_err:
                print(f"Error removing file {output_filepath}: {rm_err}")
    # --- End Modification 1 ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets and checkpoints")
    parser.add_argument("--contents", nargs="+", default=["datasets", "checkpoints"])
    parser.add_argument("--datasets", nargs="+", default=["default"])
    parser.add_argument("--checkpoints", nargs="+", default=["all"])
    parser.add_argument("--save_dir", nargs="+", default=["./data", "./results"])
    parser.add_argument("--dataset_mode", default="default")
    args = parser.parse_args()

    # --- Argument processing (determining datasets/checkpoints lists) remains the same ---
    if args.datasets[0] == "default":
        args.datasets = ["mnist", "cifar-10", "cifar-100"]  # Example default benchmarks
    elif args.datasets[0] == "ood_v1.5":
        # Define datasets for v1.5 benchmark based on the project's definition
        args.datasets = [
            "cifar-10",
            "cifar-100",
            "imagenet-200",
            "imagenet-1k",
        ]  # Adjust as needed
    elif args.datasets[0] == "all":
        args.datasets = list(benchmarks_dict.keys())  # All defined benchmarks

    if args.checkpoints[0] == "ood":
        args.checkpoints = [
            "mnist_lenet",
            "cifar10_res18",
            "cifar100_res18",
            "imagenet_res50",
        ]  # Example checkpoints
    elif args.checkpoints[0] == "ood_v1.5":
        args.checkpoints = [
            "cifar10_res18_v1.5",
            "cifar100_res18_v1.5",
            "imagenet200_res18_v1.5",
            "imagenet_res50_v1.5",
        ]  # Example v1.5 checkpoints
    elif args.checkpoints[0] == "all":
        # Define 'all' checkpoints based on download_id_dict keys (filter as needed)
        args.checkpoints = [
            k
            for k in download_id_dict
            if k not in ["benchmark_imglist"]
            and k not in benchmarks_dict
            and not any(k in v for v in benchmarks_dict.values())
        ]
        print(
            f"Selected 'all' checkpoints: {args.checkpoints}"
        )  # Check what 'all' resolved to
    # --- End Argument processing ---

    # Ensure base save directories exist
    if not os.path.exists(args.save_dir[0]):
        print(f"Creating base data directory: {args.save_dir[0]}")
        os.makedirs(args.save_dir[0])
    if len(args.save_dir) > 1 and not os.path.exists(args.save_dir[1]):
        print(f"Creating base results directory: {args.save_dir[1]}")
        os.makedirs(args.save_dir[1])

    for content in args.contents:
        if content == "datasets":
            print("\n--- Processing Datasets ---")
            # Base path for datasets, e.g., './data/'
            store_path_base = args.save_dir[0]

            # --- MODIFICATION 2: Handle benchmark_imglist download ---
            benchmark_imglist_dir = os.path.join(store_path_base, "benchmark_imglist")
            benchmark_imglist_zip = os.path.join(
                store_path_base, "benchmark_imglist.zip"
            )

            # Check if the *extracted directory* exists and is non-empty
            if os.path.exists(benchmark_imglist_dir) and any(
                os.scandir(benchmark_imglist_dir)
            ):
                print(
                    f"Directory '{benchmark_imglist_dir}' exists and is not empty, skipping download."
                )
            else:
                print(f"Directory '{benchmark_imglist_dir}' not found or empty.")
                # Check if the zip file exists (maybe download failed before extraction)
                if os.path.exists(benchmark_imglist_zip):
                    print(
                        f"Found existing zip file: {benchmark_imglist_zip}. Attempting to extract..."
                    )
                    try:
                        with zipfile.ZipFile(benchmark_imglist_zip, "r") as zip_file:
                            zip_file.extractall(store_path_base)  # Extract TO ./data/
                        print(
                            f"Extraction successful. Removing {benchmark_imglist_zip}..."
                        )
                        os.remove(benchmark_imglist_zip)
                    except Exception as e:
                        print(
                            f"ERROR extracting existing {benchmark_imglist_zip}: {e}. Deleting zip and retrying download."
                        )
                        try:
                            os.remove(benchmark_imglist_zip)
                        except OSError as rm_err:
                            print(f"Error removing {benchmark_imglist_zip}: {rm_err}")
                        # Force redownload by falling through
                # If extracted dir doesn't exist and zip doesn't (or failed extraction), download.
                if not (
                    os.path.exists(benchmark_imglist_dir)
                    and any(os.scandir(benchmark_imglist_dir))
                ):
                    output_filepath = (
                        benchmark_imglist_zip  # Full path for the zip file
                    )

                    print(
                        f"Attempting to download benchmark_imglist to {output_filepath}..."
                    )
                    try:
                        # Pass the full filepath to gdown
                        gdown.download(
                            id=download_id_dict["benchmark_imglist"],
                            output=output_filepath,
                            resume=True,
                            quiet=False,
                        )
                        # Check if download completed successfully before unzipping
                        if os.path.exists(output_filepath):
                            print(
                                f"Download successful. Unzipping {output_filepath}..."
                            )
                            with zipfile.ZipFile(output_filepath, "r") as zip_file:
                                zip_file.extractall(
                                    store_path_base
                                )  # Extract TO ./data/
                            print(
                                f"Unzipping successful. Removing {output_filepath}..."
                            )
                            os.remove(output_filepath)  # Remove the zip file
                            print(
                                "benchmark_imglist successfully downloaded and extracted."
                            )
                        else:
                            print(
                                f"ERROR: Download failed, {output_filepath} not found after gdown call."
                            )
                    except zipfile.BadZipFile:
                        print(
                            f"ERROR: Bad zip file downloaded: {output_filepath}. Deleting."
                        )
                        if os.path.exists(output_filepath):
                            os.remove(output_filepath)
                    except Exception as e:
                        print(
                            f"ERROR during benchmark_imglist download or processing: {e}"
                        )
                        if os.path.exists(output_filepath):
                            try:
                                os.remove(output_filepath)
                                print(
                                    f"Removed potentially corrupt file: {output_filepath}"
                                )
                            except OSError as rm_err:
                                print(
                                    f"Error removing file {output_filepath}: {rm_err}"
                                )
            # --- End Modification 2 ---

            # Process actual datasets based on mode
            datasets_to_download = set()
            if args.dataset_mode == "default" or args.dataset_mode == "benchmark":
                print(f"Processing benchmarks: {args.datasets}")
                for benchmark in args.datasets:
                    if benchmark in benchmarks_dict:
                        for dataset in benchmarks_dict[benchmark]:
                            datasets_to_download.add(dataset)
                    else:
                        print(
                            f"Warning: Benchmark key '{benchmark}' not found in benchmarks_dict."
                        )
            elif args.dataset_mode == "dataset":
                print(f"Processing specific datasets: {args.datasets}")
                for dataset in args.datasets:
                    datasets_to_download.add(
                        dataset
                    )  # Assume direct dataset names passed

            print(
                f"Final list of datasets to download: {sorted(list(datasets_to_download))}"
            )
            for dataset in sorted(list(datasets_to_download)):
                if dataset in download_id_dict:
                    print(f"\nProcessing dataset: {dataset}")
                    download_dataset(dataset, args)
                else:
                    print(f"Warning: Skipping '{dataset}', no download ID found.")

        elif content == "checkpoints":
            print("\n--- Processing Checkpoints ---")
            # Determine base path for checkpoints
            if (
                "v1.5" in args.checkpoints[0]
            ):  # Heuristic based on first checkpoint name
                store_path_base = args.save_dir[1]  # e.g., './results/'
                checkpoint_subdir = (
                    ""  # Save directly in results for v1.5? Adjust if needed.
                )
            else:
                store_path_base = args.save_dir[1]  # e.g., './results/'
                checkpoint_subdir = "checkpoints"  # e.g., './results/checkpoints/'

            # store_path is the directory where the ZIP should be extracted TO
            store_path = os.path.join(store_path_base, checkpoint_subdir)
            if not os.path.exists(store_path):
                print(f"Creating directory: {store_path}")
                os.makedirs(store_path)

            print(f"Processing checkpoints: {args.checkpoints}")
            for checkpoint in args.checkpoints:
                if checkpoint not in download_id_dict:
                    print(
                        f"Warning: Checkpoint '{checkpoint}' not found in download_id_dict, skipping."
                    )
                    continue

                print(f"\nProcessing checkpoint: {checkpoint}")

                # --- MODIFICATION 3: Construct full filepath and call gdown correctly ---
                expected_zip_name = checkpoint + ".zip"
                output_filepath = os.path.join(
                    store_path, expected_zip_name
                )  # Full path for the zip file

                # Check if the *extracted file/folder* seems to exist (heuristic)
                # A better check might look for specific file extensions like .pth, .pt, .ckpt
                checkpoint_extracted_path = os.path.join(
                    store_path, checkpoint
                )  # Assumes zip extracts to folder named 'checkpoint'
                if os.path.exists(checkpoint_extracted_path):
                    print(
                        f"Extracted checkpoint '{checkpoint_extracted_path}' found, skipping download."
                    )
                    continue  # Skip if extracted content seems present

                # If not extracted, proceed with download/extraction
                print(f"Attempting to download {checkpoint} to {output_filepath}...")
                try:
                    # Pass the full filepath to gdown
                    gdown.download(
                        id=download_id_dict[checkpoint],
                        output=output_filepath,
                        resume=True,
                        quiet=False,
                    )
                    # Check if download completed successfully before unzipping
                    if os.path.exists(output_filepath):
                        print(f"Download successful. Unzipping {output_filepath}...")
                        with zipfile.ZipFile(output_filepath, "r") as zip_file:
                            zip_file.extractall(
                                store_path
                            )  # Extract TO the containing directory
                        print(f"Unzipping successful. Removing {output_filepath}...")
                        os.remove(
                            output_filepath
                        )  # Remove the zip file after extraction
                        print(
                            f"Checkpoint '{checkpoint}' successfully downloaded and extracted to {store_path}."
                        )
                    else:
                        print(
                            f"ERROR: Download failed, {output_filepath} not found after gdown call."
                        )
                except zipfile.BadZipFile:
                    print(
                        f"ERROR: Bad zip file downloaded: {output_filepath}. Deleting."
                    )
                    if os.path.exists(output_filepath):
                        os.remove(output_filepath)
                except Exception as e:
                    print(f"ERROR during {checkpoint} download or processing: {e}")
                    if os.path.exists(output_filepath):
                        try:
                            os.remove(output_filepath)
                            print(
                                f"Removed potentially corrupt file: {output_filepath}"
                            )
                        except OSError as rm_err:
                            print(f"Error removing file {output_filepath}: {rm_err}")
                # --- End Modification 3 ---

    print("\n--- Download script finished ---")
