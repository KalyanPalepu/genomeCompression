import vcf

#parse de-compressed VCF file -- thousand.vcf
#then generate gaps and bases for compressor to use
#generate verify file that has SNP and Indel's locations, REF and ALT. This can be used to verify compress/uncompress result
#generate probability file distances that has the probabilty for each gap/SNP type. Can be used for probability based compression

#Iput: entries specifies how many entries need to be generated, at least 2000
def prep(entries):
    #get at least 2000 entries to get meaningful stats
    if(entries<2000):
        entries = 2000
    vcf_reader = vcf.Reader(open('thousand.vcf', 'r'))

    count = 0
    size_err = 0
    dist_overflow = 0
    snp_stats = {'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CG':0, 'CT':0,
             'GA':0, 'GC':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'OT':0} #OT for all other types not listed
    dist = [] #all distances beween two neighboring SNPs
    for x in range(100000):
        dist.append({'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CG':0, 'CT':0,
             'GA':0, 'GC':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'OT':0})

    bases = open("bases",'w')  #only alt base
    gaps = open("gaps", 'w')   #all gaps, except the first one is the absolute location
    bases2 = open("bases2", 'w') #both ref and alt bases, 0 for indel
    verify = open("verify", 'w')  #will be used to compare the final output after un-compress
    y    = [0 for x in range(1000)] #y is used for drawing
    pre_pos = 0

    for record in vcf_reader:
        try:
            size1 = len(record.REF)
            size2 = len((record.ALT[0]))
        except TypeError:
            size_err = size_err+1
            #print record.CHROM, record.POS, record.REF, record.ALT
            continue
        count = count+1
        if(count%5000 == 0):
            print record.CHROM, record.POS, record.REF, record.ALT, len(record.REF), len(record.ALT[0])
        if( (len(record.REF) == 1) and (len((record.ALT[0])) == 1)): #counting the polymorphism in the dictionary
            key = record.REF + str(record.ALT[0])
            snp_stats[key] = snp_stats[key]+1
            alt_base = str(record.ALT[0])
            ref_alt_base = key
        else:                                                        #If it is odd, it goes to OT (other)
            key = 'OT'
            snp_stats['OT'] = snp_stats['OT']+1
            alt_base = '0'
            ref_alt_base = '0'

        if((record.POS > pre_pos) and (record.POS-pre_pos) < 100000):
            idx = record.POS-pre_pos
            dist[idx][key] = dist[idx][key]+1
            pre_pos = record.POS
        else:
            dist_overflow = dist_overflow + 1
            count = count - 1
            pre_pos = record.POS

        #Writew all files, used by compress, verifications
        if key != 'OT': #skip Indels before we have a good way to handle it
            bases.write(alt_base)
            bases.write("\n")
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

        if(count > entries):
            for key in snp_stats:
                print key, count, float(snp_stats[key])/float(count)
            #gaps(distances) stats: percent for each gap, each SNP type
            fd = open("distances", 'w')
            for key in snp_stats:
                fd.write("{0:>9s}".format(key))
            fd.write("\n")

            for snp_dist in dist:
                for key in snp_stats:
                    fd.write("{0:9d}".format(snp_dist[key]))
                fd.write("\n")
            fd.close()

            bases.close()
            gaps.close()
            bases2.close()
            verify.close()
            break

#def main(_):

if __name__ == '__main__':
    prep(1500000)


