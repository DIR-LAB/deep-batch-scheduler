import os

batches = [0, 10000, 10000, 0]
len = 1024
iter = 10

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--score_type', type=str, default="bsld")
    args = parser.parse_args()
    if args.score_type == "bsld":
        dire = "trained_models/bsld/"
        workloads = ["data/lublin_256.swf", "data/SDSC-SP2-1998-4.2-cln.swf", "data/HPC2N-2002-2.2-cln.swf",
                     "data/lublin_256_new2"]
        models = ["lublin256", "sdsc_sp2", "hpc2n", "Lublin256new"]
        score_type = 0
        seed = 1
    elif args.score_type == "utilization":
        dire = "trained_models/utilization/"
        workloads = ["data/lublin_256.swf", "data/SDSC-SP2-1998-4.2-cln.swf", "data/HPC2N-2002-2.2-cln.swf",
                     "data/lublin_256_new2"]
        models = ["lublin256", "sdsc_sp2", "hpc2n", "Lublin256new"]
        score_type = 3
        seed = 1
    else:
        raise NotImplementedError

    for backfil in [0, 1]:
        for model, workload, batch_job_slice in zip(models, workloads, batches):
            print("*"*20+model+"_seed_"+str(seed)+"*"*20)
            sub_file = os.listdir(dire+"/"+model)[-1]
            command = "--rlmodel {6}{0}/{8}/ --seed {1} --len {2} --backfil {3} --score_type {4} --batch_job_slice {5} --workload {7} --iter {9}"\
                .format(model, seed, len, backfil, score_type, batch_job_slice, dire, workload, sub_file, iter)
            print(command)
            s = os.popen("python"+ " -W ignore compare-make-table.py " + command).read()

            print(s)
            print("*"*50)
