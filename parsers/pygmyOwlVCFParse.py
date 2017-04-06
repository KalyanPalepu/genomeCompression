import vcf  #need to export PYTHONPATH=/home/andrew/PyVCF
import os.path
import os
import time

# parse de-compressed VCF file
# then generate gaps and bases for compressor to use
# generate verify file that has SNP and Indel's locations, REF and ALT. This can be used to verify compress/uncompress result
# generate probability file distances that has the probabilty for each gap/SNP type. Can be used for probability based compression

# Input: entries specifies how many entries need to be generated, at least 2000
def prep():
    start_time = time.time()
    filename = raw_input("The VCF file name to be parsed: ") #'3QUAQIE-EXT.vcf'
    genome = raw_input('The name of the Genome to be parsed:') #''3QUAQIE-EXT'
    directory = os.path.dirname("/home/andrew/data/" + genome + "/")
    #filename = raw_input("What is the name of the VCF file?")
    vcf_reader = vcf.Reader(open(filename, 'r'))

    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print "Genome ", genome, "is parsed already. To parse again, delete the directory first"
        os._exit(2)

    count = 0
    size_err = 0
    gaps_overflow = 0

    gapsStats = [i for i in range(100000)]
    allSNPS   = ['AC', 'AT', 'AG', 'CA', 'CT', 'CG', 'TA', 'TC', 'TG', 'GA', 'GC', 'GT']
    curChr = ''
    snpsAndGaps = None
    gaps        = None
    bases2      = None
    verify      = None
    for record in vcf_reader:
        try:
            size1 = len(record.REF)
            size2 = len((record.ALT[0]))
        except TypeError:
            size_err = size_err+1
            #print record.CHROM, record.POS, record.REF, record.ALT
            continue
        count = count+1
        if(count%10000 == 0):
            print record.CHROM, record.POS, record.REF, record.ALT, len(record.REF), len(record.ALT[0])

        #if we find a new chrom, start new files for gaps, bases,
        if record.CHROM != curChr:
            print curChr
            if gaps != None:
                gaps.close()
                bases2.close()
                verify.close()

            curChr = record.CHROM
            gapsF = "gaps" + curChr
            bases2F = "bases2" + curChr
            verifyF = "verify" + curChr

            gaps = open(os.path.join(directory, gapsF), 'w')
            bases2 = open(os.path.join(directory, bases2F), 'w')
            verify = open(os.path.join(directory, verifyF), 'w')
            pre_pos = 0

        if( (len(record.REF) == 1) and (len((record.ALT[0])) == 1)):
            key = record.REF + str(record.ALT[0])
            if key in allSNPS:
                alt_base = str(record.ALT[0])
                ref_alt_base = key
            else:
                key = '0'
                alt_base = '0'
                ref_alt_base = '0'
        else:
            key = '0'
            alt_base = '0'
            ref_alt_base = '0'
        if((record.POS > pre_pos) and (record.POS-pre_pos) < 100000):
            idx = record.POS-pre_pos
            gapsStats[idx] += 1
            pre_pos = int(record.POS)
        else:
            idx = int(record.POS) - int(pre_pos)
            gaps_overflow = gaps_overflow + 1
            pre_pos = int(record.POS)

        #Writew all files, used by compress, verifications
        if key in allSNPS: #skip Indels before we have a good way to handle it
            bases2.write(ref_alt_base)
            bases2.write("\n")

        gaps.write(str(idx))
        gaps.write("\n")

        all_alts = ''
        for alt in record.ALT:
            all_alts = all_alts + str(alt)

        a_verify_line = str(record.POS)+','+str(record.REF)+','+ all_alts #a line in the verify file
        verify.write(a_verify_line)
        verify.write("\n")

    gaps.close()
    bases2.close()
    verify.close()
    print "Parsed SNPs: ", count, "and seconds taken: ", time.time()-start_time, "gaps over 100000: ", gaps_overflow
#def main(_):

if __name__ == '__main__':
    prep()


