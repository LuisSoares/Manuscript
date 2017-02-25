
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import h5py 
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr as spearman
from scipy.ndimage.filters import maximum_filter
from numpy import argwhere
import collections


mpl.rcParams['pdf.fonttype'] = 42 #to make pdf text editable


class Gene():
    '''
    Creates a gene class with instances being the simplest description of a gene that should contain
    its name, chromossome, star location, end location and strand
    
    arguments: name - string corresponding to the name of the gene
               chromosome - string containing the roman numeral of the chromosome where the gene is located
               start - int corresponding to the lowest value location of the gene transcript
               end - int corresponding to the highest value location of the gene transcript
               strand - string corresponding to the strand of the gene, should be '+' or '-'
               
    methods: repr - returns a string containing formated information about the gene instance 
    '''
    def __init__(self,name,chromosome,start,end,strand):
        self.name=name
        self.chromosome=chromosome
        self.start=int(start)
        self.end=int(end)
        self.strand=strand
    def __repr__(self):
        return "Gene name:{} ,Start:{}, End:{}, Chromosome:{}, Strand{}".format(self.name,self.start,self.end,
                                                                                self.chromosome,self.strand)

def create_gene_list(bed):
    '''Retuns a list of gene instances
    
    Takes a bed file and creates a list containing gene class instances
    arguments: bed - string containing the path to a bed file describing genes as intervals
    '''
    gene_list=[] # this will be the gene list we will populate
    with open(bed) as data:
        data.readline() #get rid of the first line since is an header
        for line in data:
            temp=line.split('\t')
            """Two things to notice, first Start is always smaller than End meaning that the names
            don't reflect the biological meaning, this is important because they should be
            interpreted together with the strand value, second the chromosome location uses 
            the chrX notation in the burak file which is unlike the description in the dataset we are going
            to use which only uses X for chromosome names, for that reason the file chromosome location
            description is sliced on the index 3 (to remove the 'chr' part)
            """
            gene_list.append(Gene(temp[4].rstrip(),temp[0][3:],temp[1],temp[2],temp[3]))
    return gene_list

def create_hdf5_track(dataset_name,hdf5_file_handler):
    """Returns None
    
    Takes a dataset_name and a hdf5 file handler and creates hdf5 group with the dataset name, the group
    will contain subgroups with the chromosome names and empty arrays with the chromosome sizes
    
    arguments: dataset_name - string with the name of the dataset
               hdf5_file_handler - handle to open hdf5 file
    """
    
    list_of_chromosomes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
                          'XVI', 'Mito'] # The names of all chromosomes
    size_of_chromosomes = [230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643, 439888, 745751, 666816,
                           1078177, 924431, 784333, 1091291, 948066, 85779] #the sizes of all chromosomes
    chromosomes=[(chrom,size) for chrom,size in zip(list_of_chromosomes,size_of_chromosomes)]
    temp=hdf5_file_handler.create_group(dataset_name) #create a first group with the name of our dataset
    for item in chromosomes:
        temp.create_dataset(item[0],(item[1],),dtype='f') #to the dataset group we add subgroups containing empty arrays

def load_track_in_hdf5(file,name,hdf5,file_type,norm=1):
    """Returns None
    
    Populates and hdf5 structure previously built with create_hdf5_dataset function with data from wig file 
    according to a dataset name
    
    arguments: bdg - string containing the path to a beg file
               name - string containing the name of a previously created hdf5 subgroup
               hdf5 - previously openned hdf5 file handler
               """
    if file_type=='bed':
        data=load_bed(file,norm)
    elif file_type=='wig':
        data=load_wig(file)

    for key,value in data.items():
            hdf5[name][key][...] = value #populate hdf5 file

def load_wig(wig):
    """
    returns dictionary of chromosomes
    
    loads wig file in a dictionary of numpy arrays takes as input a wig file with 1 nucleotide steps
    outputs a dictionary with keys being chromosome names and values numpy arrays with data (chromosome names 
    should be roman numerals)
    
    arguments: wig - string containing path to wig file
    
    """
    
    list_of_chromosomes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
                          'XVI'] # The names of all chromosomes
    size_of_chromosomes = [230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643, 439888, 745751, 666816,
                           1078177, 924431, 784333, 1091291, 948066] #the sizes of all chromosomes
    chromosomes={chrom:np.zeros(size) for chrom,size in zip(list_of_chromosomes,size_of_chromosomes)}
    with open(wig) as data:
        data.readline()
        for line in data:
            if line[0]=='v':
                current_chromosome=line.split('=')[1][3:].strip()
                #print(current_chromosome)
            else:
                coord,value=line.rstrip().split('\t')
                coord=int(coord)
                value=float(value)
                chromosomes[current_chromosome][coord-1]=value
    return chromosomes

def load_bed(bed,norm=1):
    """
    returns dictionary of chromosomes
    
    loads bdg file in a dictionary of numpy arrays takes as input a bdg file with 1 nucleotide steps
    outputs a dictionary with keys being chromosome names and values numpy arrays with data (chromosome names 
    should be roman numerals)
    
    arguments: bed - string containing path to bdg file
    
    """
    
    list_of_chromosomes = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV',
                          'XVI', 'Mito'] # The names of all chromosomes
    size_of_chromosomes = [230218, 813184, 316620, 1531933, 576874, 270161, 1090940, 562643, 439888, 745751, 666816,
                           1078177, 924431, 784333, 1091291, 948066, 85779] #the sizes of all chromosomes
    chromosomes={chrom:np.zeros(size) for chrom,size in zip(list_of_chromosomes,size_of_chromosomes)}
    with open(bed) as data:
        for line in data:
            chrom,coord_start,coord_end,value=line.rstrip().split('\t')
            coord_start=int(coord_start)
            coord_end=int(coord_end)
            value=float(value)*norm
            chromosomes[chrom[3:]][coord_start:coord_end]=value
    return chromosomes
	


def create_anchor_plot(hdf5_file,track,start,end,genes):
    array=np.zeros((7000,(end-start)))
    number_of_plotted=0
    number_not_plotted_outside=0
    number_not_plotted_zeros=0
    genes_plotted=[]
    for item in genes:
        if item.strand=='+':
            try:
                current_array=hdf5_file[track][item.chromosome][item.start+start:item.start+end]
                array[number_of_plotted]=current_array
                number_of_plotted+=1
                genes_plotted.append(item)
            except:
                number_not_plotted_outside+=1
                print(item.name)
        if item.strand=='-':
            try:
                current_array=hdf5_file[track][item.chromosome][item.end-end:item.end-start][::-1]
                array[number_of_plotted]=current_array
                number_of_plotted+=1
                genes_plotted.append(item)
            except:
                number_not_plotted_outside+=1
                print(item.name)
                
    return array[:number_of_plotted,:],number_of_plotted,number_not_plotted_outside,number_not_plotted_zeros,genes_plotted
        

def create_array(hdf5_file,dataset,func,genes):
    array=[]
    for item in genes:
        if item.strand=='+':
            temp=hdf5_file[dataset][item.chromosome][item.start:item.end]
            array.append(func(temp))
        elif item.strand=='-':
            temp=hdf5_file[dataset][item.chromosome][item.start:item.end][::-1]
            array.append(func(temp))
    return array

def create_max_array(hdf5_file,dataset,genes,span):
    array=[]
    for item in genes:
        if abs(item.end-item.start)<span:
            temp=np.max(hdf5_file[dataset][item.chromosome][item.start:item.end])
            array.append(temp)
        else:
            if item.strand=='+':
                temp=hdf5_file[dataset][item.chromosome][item.start:item.start+span]
                array.append(np.max(temp))
            elif item.strand=='-':
                temp=hdf5_file[dataset][item.chromosome][item.end-span:item.end][::-1]
                array.append(np.max(temp))
    return array

def create_argmax_array(hdf5_file,dataset,genes,span):
    array=[]
    for item in genes:
        if abs(item.end-item.start)<span:
            temp=np.argmax(hdf5_file[dataset][item.chromosome][item.start:item.end])
            array.append(temp)
        else:
            if item.strand=='+':
                temp=hdf5_file[dataset][item.chromosome][item.start:item.start+span]
                array.append(np.argmax(temp))
            elif item.strand=='-':
                temp=hdf5_file[dataset][item.chromosome][item.end-span:item.end][::-1]
                array.append(np.argmax(temp))
    return array


def findPeaks(data,threshold,size=100,mode='constant'):
    peaks=[]
    data=np.array(data)
    if (len(data)==0 or np.max(data)<threshold):
        return peaks
    boolsVal=data>threshold
    maxFilter=maximum_filter(data,size=size,mode=mode)
    boolsMax=data==maxFilter
    boolsPeak= boolsVal & boolsMax
    indices=argwhere(boolsPeak)
    for position in indices:
        position=position
        height=data[position]
        peak=(position,height)
        if not (peak[1] in [item[1] for item in peaks]):
            peaks.append(peak)
    return peaks

