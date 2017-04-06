# parser for James W Genome only.  It is not the VCF format, but pre-parsed format similar to VCF
# The souce file is: /home/andrew/JWData/files/JWBU
# The output is the SNP info for each chromsome, at directory /home/andrew/data/JW/
# InDels ignored

import os.path
import os
import time

def prep(entries, genome = "JW"):
    start_time = time.time()
    filename = '/home/andrew/JWData/files/JWBU'
    vcf_reader = open(filename, 'r')

    count = 0
    size_err = 0
    dist_overflow = 0

    directory = os.path.dirname("/home/andrew/data/"+genome+"/")
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print "Genome ", genome, "is parsed already. To parse again, delete the directory first"
        os._exit(2)

    curChr = "chr1"
    sAndGF = "sAndG"+curChr
    gapsF  = "gaps"+curChr
    bases2F = "bases2"+curChr
    snpsAndGaps = open(os.path.join(directory, sAndGF), 'w') #both ref and alt bases, 0 for indel
    gaps = open(os.path.join(directory, gapsF), 'w')
    bases2 = open(os.path.join(directory, bases2F), 'w')
    pre_pos = 0
    #Start from chr1
    #loop through all chroms
    #Put SNPs, GAPS, SNP and GAPS to its own file for compressor to use

    for line in vcf_reader:
        line1 = line.strip()
        sample, chr, loc, snp = line1.split(',')
        if chr != curChr:
            sAndGF = "sAndG" + chr
            if os.path.isfile(os.path.join(directory, sAndGF)):
                continue
            else:
                snpsAndGaps.close()
                gaps.close()
                bases2.close()
                curChr = chr
                print chr, snp
                pre_pos = 0
                sAndGF = "sAndG" + curChr
                gapsF = "gaps" + curChr
                bases2F = "bases2" + curChr
                snpsAndGaps = open(os.path.join(directory, sAndGF), 'w')  # both ref and alt bases, 0 for indel
                gaps = open(os.path.join(directory, gapsF), 'w')
                bases2 = open(os.path.join(directory, bases2F), 'w')

        ref, alt = snp.split('/')

        pos = int(loc)
        gap = pos - pre_pos
        try:
            size1 = len(ref)
            size2 = len(alt)
        except TypeError:
            size_err = size_err+1
            #print record.CHROM, record.POS, record.REF, record.ALT
            continue
        count = count+1

        if size1 == 1 and size2 == 1:
            snpsAndGaps.write('{0},{1},{2} \n'.format(gap, ref, alt))
            key = ref[0]+alt[0]
            bases2.write('{0}\n'.format(key))
        gaps.write('{0}\n'.format(gap))
        pre_pos = pos
        if count>=entries:
            break
    vcf_reader.close()
    snpsAndGaps.close()
    gaps.close()
    bases2.close()
    print "Parsed SNPs:", count, "with seconds taken: ", time.time()-start_time

#def main(_):

if __name__ == '__main__':
    genome_to_parse = "JW" #The genome is JW
    prep(18000000, genome_to_parse)


