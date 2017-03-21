import vcf

# pygme_pre with all of the stat calculations removed (to speed up parsing)


def prep(entries):
    filename = raw_input("What is the name of the VCF file?")
    vcf_reader = vcf.Reader(open(filename, 'r'))

    count = 0
    gaps = open("hugeData/gaps.g", 'w')   # all gaps, except the first one is the absolute location
    bases2 = open("hugeData/bases2.b", 'w')  # both ref and alt bases, 0 for indel
    pre_pos = 0

    progressMarker = entries / 100
    progressMarked = 0

    for record in vcf_reader:
        if count % progressMarker == 0:
            print "Done with {0}% of file".format(progressMarked)

        try:
            size1 = len(record.REF)
            size2 = len((record.ALT[0]))
        except TypeError:
            continue

        count += 1

        if len(record.REF) == 1 and len(record.ALT[0]) == 1:  # counting the polymorphism in the dictionary
            key = record.REF + str(record.ALT[0])
        else:                                                        #If it is odd, it goes to OT (other)
            key = 'OT'

        idx = record.POS - pre_pos
        # if record.POS > pre_pos and record.POS-pre_pos < 100000:
        #     pre_pos = record.POS
        # else:
        #     print "This should never happen"
        #     count -= 1
        #     pre_pos = record.POS

        # Write all files, used by compress, verifications
        if key != 'OT':  # skip Indels before we have a good way to handle it
            bases2.write(key + '\n')
            gaps.write(str(idx))

        if count > entries:
            break

    gaps.close()
    bases2.close()


if __name__ == '__main__':
    from timeit import default_timer
    #entries = 16000 * 6250
    entries = 16000 * 2500
    start = default_timer()
    prep(entries)
    end = default_timer()
    print "Total Time: {0}".format(end - start)


