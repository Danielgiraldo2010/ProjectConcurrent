import argparse
import numpy as np
import cv2
import time
from tqdm import tqdm
from mpi4py import MPI
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from Bio import SeqIO


def parse_fasta(file):
    seqs = [str(record.seq) for record in SeqIO.parse(file, "fasta")]
    return "".join(seqs)


def create_dotplot(image, filename='dotplot.svg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="Greys", aspect="auto")
    plt.xlabel("Sequence 1")
    plt.ylabel("Sequence 2")
    plt.savefig(filename)
    plt.show()


def sequential_dotplot(seq1, seq2):
    dp = np.zeros((len(seq1), len(seq2)))
    for i in tqdm(range(len(seq1))):
        for j in range(len(seq2)):
            dp[i, j] = 1 if seq1[i] == seq2[j] else 0.7 if i == j else 0
    return dp


def dotplot_worker(args):
    i, seq1, seq2 = args
    return [1 if seq1[i] == seq2[j] else 0.7 if i == j else 0 for j in range(len(seq2))]


def multiprocessing_dotplot(seq1, seq2, num_workers=cpu_count()):
    with Pool(processes=num_workers) as pool:
        dp = pool.map(dotplot_worker, [(i, seq1, seq2) for i in range(len(seq1))])
    return dp


def save_results(results, filename="images/results.txt"):
    with open(filename, "w") as file:
        for result in results:
            file.write(f"{result}\n")


def calc_acceleration(times):
    return [times[0] / t for t in times]


def calc_efficiency(accels, threads):
    return [accels[i] / threads[i] for i in range(len(threads))]


def plot_multiprocessing_graphs(times, accels, effs, threads):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(threads, times)
    plt.xlabel("Number of Processors")
    plt.ylabel("Time")
    plt.subplot(1, 2, 2)
    plt.plot(threads, accels, label="Acceleration")
    plt.plot(threads, effs, label="Efficiency")
    plt.xlabel("Number of Processors")
    plt.ylabel("Acceleration and Efficiency")
    plt.legend()
    plt.savefig("images/images_multiprocessing/graphs_multiprocessing.png")


def plot_mpi_graphs(times, accels, effs, threads):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(threads, times)
    plt.xlabel("Number of Processors")
    plt.ylabel("Time")
    plt.subplot(1, 2, 2)
    plt.plot(threads, accels, label="Acceleration")
    plt.plot(threads, effs, label="Efficiency")
    plt.xlabel("Number of Processors")
    plt.ylabel("Acceleration and Efficiency")
    plt.legend()
    plt.savefig("images/images_mpi/graphs_mpi.png")


def mpi_dotplot(seq1, seq2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    partitions = np.array_split(range(len(seq1)), size)
    dp = np.zeros((len(partitions[rank]), len(seq2)), dtype=np.float16)
    
    for i in tqdm(range(len(partitions[rank]))):
        for j in range(len(seq2)):
            dp[i, j] = 1.0 if seq1[partitions[rank][i]] == seq2[j] else 0.6 if i == j else 0.0

    dp = comm.gather(dp, root=0)
    
    if rank == 0:
        final_dp = np.vstack(dp)
        return final_dp


def apply_image_filter(matrix, output_image):
    kernel = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    filtered = cv2.filter2D(matrix, -1, kernel)
    normalized = cv2.normalize(filtered, None, 0, 127, cv2.NORM_MINMAX)
    
    _, thresholded = cv2.threshold(normalized, 50, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(output_image, thresholded)
    cv2.imshow('Filtered Image', thresholded)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, required=True, help='First sequence file in FASTA format')
    parser.add_argument('--file2', type=str, required=True, help='Second sequence file in FASTA format')
    parser.add_argument('--sequential', action='store_true', help='Run in sequential mode')
    parser.add_argument('--multiprocessing', action='store_true', help='Run with multiprocessing')
    parser.add_argument('--mpi', action='store_true', help='Run with MPI')
    parser.add_argument('--num_processes', type=int, nargs='+', default=[4], help='Number of processes for MPI')
    
    args = parser.parse_args()
    
    if rank == 0:
        start_load = time.time()
        seq1 = parse_fasta(args.file1)
        seq2 = parse_fasta(args.file2)
        seq1 = seq1[:16000]
        seq2 = seq2[:16000]
        end_load = time.time()
        
        save_results([f"File loading time: {end_load - start_load}"], "images/results_load.txt")
        
        dotplot = np.zeros((len(seq1), len(seq2)))
        results = []
        mpi_results = []
        multiprocessing_times = []
        mpi_times = []
    
    if args.sequential:
        start_seq = time.time()
        seq_dotplot = sequential_dotplot(seq1, seq2)
        results.append(f"Sequential execution time: {time.time() - start_seq}")
        create_dotplot(seq_dotplot[:600, :600], "images/images_sequential/dotplot_sequential.png")
        apply_image_filter(seq_dotplot[:600, :600], 'images/images_filter/dotplot_filter_sequential.png')
        save_results(results, "images/results_sequential.txt")
    
    if args.multiprocessing:
        thread_counts = [1, 2, 4, 8]
        for thread_count in thread_counts:
            start_time = time.time()
            mp_dotplot = np.array(multiprocessing_dotplot(seq1, seq2, thread_count))
            multiprocessing_times.append(time.time() - start_time)
            results.append(f"Multiprocessing execution time with {thread_count} threads: {time.time() - start_time}")
        
        accelerations = calc_acceleration(multiprocessing_times)
        for i, accel in enumerate(accelerations):
            results.append(f"Acceleration with {thread_counts[i]} threads: {accel}")
        
        efficiencies = calc_efficiency(accelerations, thread_counts)
        for i, eff in enumerate(efficiencies):
            results.append(f"Efficiency with {thread_counts[i]} threads: {eff}")
        
        save_results(results, "images/results_multiprocessing.txt")
        plot_multiprocessing_graphs(multiprocessing_times, accelerations, efficiencies, thread_counts)
        create_dotplot(mp_dotplot[:600, :600], 'images/images_multiprocessing/dotplot_multiprocessing.png')
        apply_image_filter(mp_dotplot[:600, :600], 'images/images_filter/dotplot_filter_multiprocessing.png')
    
    if args.mpi:
        num_threads = args.num_processes
        for num_thread in num_threads:
            start_time = time.time()
            mpi_dotplot_result = mpi_dotplot(seq1, seq2)
            mpi_times.append(time.time() - start_time)
            mpi_results.append(f"MPI execution time with {num_thread} threads: {time.time() - start_time}")
        
        accelerations = calc_acceleration(mpi_times)
        for i, accel in enumerate(accelerations):
            mpi_results.append(f"Acceleration with {num_threads[i]} threads: {accel}")
        
        efficiencies = calc_efficiency(accelerations, num_threads)
        for i, eff in enumerate(efficiencies):
            mpi_results.append(f"Efficiency with {num_threads[i]} threads: {eff}")
        
        save_results(mpi_results, "images/results_mpi.txt")
        plot_mpi_graphs(mpi_times, accelerations, efficiencies, num_threads)
        create_dotplot(mpi_dotplot_result[:600, :600], 'images/images_mpi/dotplot_mpi.png')
        apply_image_filter(mpi_dotplot_result[:600, :600], 'images/images_filter/dotplot_filter_mpi.png')


if __name__ == "__main__":
    main()
