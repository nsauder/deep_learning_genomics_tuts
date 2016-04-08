import numpy as np
import pickle
import h5py
import numpy as np
import os
import re
import dateutil.parser
import datetime

fX = np.float32
BASE_DIR = ""


def preproc_event(mean, std, length):
  mean = mean / 100.0 - 0.66
  std = std - 1
  return [mean, mean*mean, std, length]


def load_read_data(read_file):
  h5 = h5py.File(read_file, "r")
  ret = {}
  
  log = h5["Analyses/Basecall_2D_000/Log"][()]
  temp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling template.*", log).groups()[0])
  comp_time = dateutil.parser.parse(re.search(r"(.*) Basecalling complement.*", log).groups()[0])
  comp_end_time = dateutil.parser.parse(re.search(r"(.*) Aligning hairpin.*", log).groups()[0])

  start_2d_time = dateutil.parser.parse(re.search(r"(.*) Performing full 2D.*", log).groups()[0])
  end_2d_time = dateutil.parser.parse(re.search(r"(.*) Workflow completed.*", log).groups()[0])

  ret["temp_time"] = comp_time - temp_time
  ret["comp_time"] = comp_end_time - comp_time
  ret["2d_time"] = end_2d_time - start_2d_time

  try:
    ret["called_template"] = h5["Analyses/Basecall_2D_000/BaseCalled_template/Fastq"][()].split('\n')[1]
    ret["called_complement"] = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Fastq"][()].split('\n')[1]
    ret["called_2d"] = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Fastq"][()].split('\n')[1]
  except Exception as e:
    print "wat", e 
    return None
  events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  ret["mp_template"] = []
  for e in events:
    if e["move"] == 1:
      ret["mp_template"].append(e["mp_state"][2])
    if e["move"] == 2:
      ret["mp_template"].append(e["mp_state"][1:3])
  ret["mp_template"] = "".join(ret["mp_template"])
  tscale = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["scale"]
  tscale_sd = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["scale_sd"]
  tshift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["shift"]
  tdrift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_template"].attrs["drift"]
  index = 0.0
  ret["temp_events"] = []
  for e in events:
    mean = (e["mean"] - tshift - index * tdrift) / tscale
    stdv = e["stdv"] / tscale_sd
    length = e["length"]
    ret["temp_events"].append(preproc_event(mean, stdv, length))
    index += e["length"]
  events = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Events"]
  cscale = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["scale"]
  cscale_sd = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["scale_sd"]
  cshift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["shift"]
  cdrift = h5["/Analyses/Basecall_2D_000/Summary/basecall_1d_complement"].attrs["drift"]
  index = 0.0
  ret["comp_events"] = []
  for e in events:
    mean = (e["mean"] - cshift - index * cdrift) / cscale
    stdv = e["stdv"] / cscale_sd
    length = e["length"]
    ret["comp_events"].append(preproc_event(mean, stdv, length))
    index += e["length"]
  ret["temp_events"] = np.array(ret["temp_events"], dtype=np.float32)
  ret["comp_events"] = np.array(ret["comp_events"], dtype=np.float32)

  al = h5["Analyses/Basecall_2D_000/BaseCalled_2D/Alignment"]
  temp_events = h5["Analyses/Basecall_2D_000/BaseCalled_template/Events"]
  comp_events = h5["Analyses/Basecall_2D_000/BaseCalled_complement/Events"]
  ret["2d_events"] = []
  for a in al:
    ev = []
    if a[0] == -1:
      ev += [0, 0, 0, 0, 0]
    else:
      e = temp_events[a[0]]
      mean = (e["mean"] - tshift - index * tdrift) / cscale
      stdv = e["stdv"] / tscale_sd
      length = e["length"]
      ev += [1] + preproc_event(mean, stdv, length)
    if a[1] == -1:
      ev += [0, 0, 0, 0, 0]
    else:
      e = comp_events[a[1]]
      mean = (e["mean"] - cshift - index * cdrift) / cscale
      stdv = e["stdv"] / cscale_sd
      length = e["length"]
      ev += [1] + preproc_event(mean, stdv, length)
    ret["2d_events"].append(ev) 
  ret["2d_events"] = np.array(ret["2d_events"], dtype=np.float32)

  h5.close()

  return ret


def one_hot_encode_gene(sequence):
    base_matrix = np.eye(4)
    nucleotides = 'ACGT'
    
    return np.array([base_matrix[nucleotides.find(bp)]
                     for bp in sequence])


def load_data(directory_name):
    inputs = []
    outputs = []
    
    for dirname, _, files in os.walk(directory_name):
        for filename in files:
            if filename.endswith("fast5"):
                data = load_read_data(os.path.join(dirname, filename))
                inputs.append(data["temp_events"][np.newaxis,...])
                outputs.append(one_hot_encode_gene(data["called_template"])[np.newaxis:])
    
    x_data = [seq.astype(fX) for seq in inputs]
    y_data = [seq.astype(fX) for seq in outputs]

    return x_data, y_data
