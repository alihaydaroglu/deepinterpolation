import numpy as n
import tensorflow.keras as keras
import tifffile
from multiprocessing import Pool
from deepinterpolation.generator_collection import DeepGenerator



class TifGenerator(DeepGenerator):
    def __init__(self,tif_paths, padded_size, batch_size, verbose=3,
                 pre_post_frame = 3, edge_buffer=20, sampling_stride=7, n_samples = None,
                 normalizer_sample_size = 100, n_procs_io = 14, shuffle=True, seed=2358):
        self.files = tif_paths
        self.nxy = padded_size
        self.batch_size = batch_size
        self.pre_post_frame = pre_post_frame
        self.edge_buffer = edge_buffer
        self.sampling_stride = sampling_stride
        self.n_files = len(self.files)
        self.normalizer_sample_size = normalizer_sample_size
        self.verbose = verbose
        self.shuffle = shuffle

        self.x_size = (self.batch_size, self.nxy, self.nxy, 2*self.pre_post_frame)
        self.y_size = (self.batch_size, self.nxy, self.nxy, 1)

        self.rand = n.random.RandomState(seed)

        self.log("Initializing files...")
        self.init_files(n_procs = n_procs_io)
        # self.log("Initialized files.")

        self.log("Initializing batches")
        self.init_samples(n_samples)
        self.shuffle_batches()
        
    def __getitem__(self, idx):
        ins = self.batches_in[idx]
        outs = self.batches_out[idx]

        batch_x = n.zeros(self.x_size, dtype='float32')
        batch_y = n.zeros(self.y_size, dtype='float32')

        self.log("Loading batch %d" % idx, level=1)
        for batch_idx in range(self.batch_size):
            file_idx, in_idxs = ins[batch_idx]
            __, out_idx = outs[batch_idx]

            self.log("Sample %d/%d" % (batch_idx+1, self.batch_size), level=2)
            self.log("File idx: %d" % file_idx, level=2)
            self.log("Input idxs:" + str(in_idxs), level=2)
            self.log("Output idxs:" + str(out_idx) + "\n", level=2)

            xs, y = load_frames_from_tif(self.files[file_idx], in_idxs, out_idx,
                                         self.means[file_idx], self.stds[file_idx], pad_to=self.nxy)
            batch_x[batch_idx] = xs
            batch_y[batch_idx] = y
        
        return batch_x, batch_y

    def __get_norm_parameters__(self, idx):
        return self.means[idx], self.stds[idx]

    def __len__(self):
        return len(self.batches_in)

    # def on_epoch_end():
    #     # Define this to be able to iterate over the entire set

    def init_files(self, n_procs = 14):
        self.n_frames = []
        self.means = []
        self.stds = []
        p = Pool(processes=n_procs)
        all_metadata = p.starmap(get_tif_metadata, [(f, self.normalizer_sample_size, self.rand) for f in self.files])
        total_n_frames = 0
        for n_f, mean, std in all_metadata:
            self.n_frames.append(n_f)
            self.means.append(mean)
            self.stds.append(std)
            total_n_frames += n_f
        self.log("Received %d files, total of %d frames" % (len(self.files), total_n_frames))

    def init_samples(self, n_samples=None):
        self.inputs = []
        self.outputs = []
        for f_idx, f in enumerate(self.files):
            ins, outs = get_in_out_idxs(self.n_frames[f_idx], file_index = f_idx,
                                        edge_buffer = self.edge_buffer, sampling_stride = self.sampling_stride,
                                        pre_post_frame = self.pre_post_frame)
            self.inputs += ins
            self.outputs += outs
        assert len(self.inputs) == len(self.outputs)
        self.n_samples = len(self.inputs)
        if n_samples is not None:
            self.n_samples = min(self.n_samples, n_samples)
        self.inputs = n.array(self.inputs, dtype='object')
        self.outputs = n.array(self.outputs, dtype='object')
        self.log("Found %d samples" % self.n_samples, level=1)
        

    def shuffle_batches(self):
        batch_size = self.batch_size
        self.n_batches = self.n_samples // batch_size

        if self.shuffle:
            sample_idxs = self.rand.choice(n.arange(self.n_samples), self.n_batches*batch_size)
        else:
            sample_idxs = n.arange(self.n_batches*batch_size)

        self.batches_in = [] 
        self.batches_out = []

        for i in range(self.n_batches):
            indices = sample_idxs[i*batch_size : (i+1)*batch_size]
            self.batches_in.append(self.inputs[indices])
            self.batches_out.append(self.outputs[indices])

        self.log("%d batches with %d samples each, using %d/%d available samples" %
                    (self.n_batches, len(indices), self.n_batches*batch_size, self.n_samples),
                    level=1)


    def log(self, str, level = 0):
        if self.verbose > level:
            print("\t"*level, str)



def load_frames_from_tif(file, in_idxs, out_idx, mean=0, std=1, pad_to=None):
    x_imgs = tifffile.imread(file, key = in_idxs)
    x_imgs = n.moveaxis(x_imgs, 0, -1)
    y_img  = tifffile.imread(file, key = (out_idx,))[:,:,n.newaxis]
    
    x_imgs = ((x_imgs - mean) / std).astype("float32")
    y_img  = ((y_img  - mean) / std).astype("float32")

    if pad_to is not None:
        ny, nx, nz = x_imgs.shape
        x_imgs = n.pad(x_imgs, ((0,pad_to-ny), (0,pad_to-nx), (0,0)))
        y_img = n.pad(y_img, ((0,pad_to-ny), (0,pad_to-nx), (0,0)))

    
    return x_imgs, y_img


def get_tif_metadata(file, normalizer_sample_size=200, random_state = None):
    '''
    Load a sample of the tif and get required metadata, and mean/std

    Args:
        file (str): path to tiffile
        normalizer_sample_size (int, optional): Number of frames to sample from the tiff to calculate mean/std. Defaults to 200.

    Returns:
        n_frames (int), 
        mean (float), 
        std (float)
    ''' 
    tif = tifffile.TiffFile(file)
    n_frames = len(tif.pages)

    if random_state is None:
        random_state = n.random
    
    normalizer_sample_size = min(normalizer_sample_size, n_frames)
    sample_frame_idxs = random_state.choice(n.arange(n_frames), 
                                        normalizer_sample_size)
    # print(sample_frame_idxs)
    tif_sample = tifffile.imread(file, key=sample_frame_idxs)
    
    mean = tif_sample.mean()
    std = tif_sample.std()
    
    return n_frames, mean, std

def get_in_out_idxs(n_frames, edge_buffer = 10, pre_post_frame = 3, sampling_stride = 7, file_index=None):
    '''
    Given the number of frames in a file, calculate the input indices and the output index of each sample.

    Args:
        n_frames (int): number of frames in tiff file
        edge_buffer: number of frames at the beginning and end of the tiffile to discard
        sampling_stride: Number of frames between each successive sample extracted
        pre_post_frame: number of frames on each side of the center frame to be used as input

    Returns:
        input_idxs (list): list of lists, each sublist is 2*pre_post_frame long and there are n_samples of them
        output_idxs(list): list of len n_samples
    '''                
    input_idxs = []
    output_idxs = []

    valid_frames = n.arange(edge_buffer, n_frames - edge_buffer)


    n_samples = len(valid_frames) // sampling_stride

    idx = 0
    for i in range(n_samples):
        start = idx
        center = idx + pre_post_frame
        fin = idx + 2*pre_post_frame+1

        output = valid_frames[center]
        inputs = n.concatenate([valid_frames[start:center]\
                                ,valid_frames[center+1:fin]])

        if file_index is None:
            input_idxs.append(inputs)
            output_idxs.append(output)
        else:
            input_idxs.append((file_index, inputs))
            output_idxs.append((file_index, output))

        idx += sampling_stride
    
    return input_idxs, output_idxs