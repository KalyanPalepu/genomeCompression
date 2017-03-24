import vcf

# pygme_pre with all of the stat calculations removed (to speed up parsing)


def prep():
    filename = raw_input("What is the name of the VCF file?")
    vcf_reader = vcf.Reader(open(filename, 'r'))

    count = 0
    gaps = open("/home/kalyanp/hugeData/gaps.g", 'w')   # all gaps, except the first one is the absolute location
    bases2 = open("/home/kalyanp/hugeData/bases2.b", 'w')  # both ref and alt bases, 0 for indel
    pre_pos = 0

    for record in vcf_reader:
        try:
            size1 = len(record.REF)
            size2 = len((record.ALT[0]))
        except TypeError:
            continue

        count += 1

        if len(record.REF) == 1 and len(record.ALT[0]) == 1:  # counting the polymorphism in the dictionary
            bases2.write(record.REF + str(record.ALT[0]) + '\n')

    gaps.close()
    bases2.close()


if __name__ == '__main__':
    from timeit import default_timer
    start = default_timer()
    prep()
    end = default_timer()
    print "Total Time: {0}".format(end - start)


