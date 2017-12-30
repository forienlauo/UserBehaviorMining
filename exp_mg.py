import logging
import os
import shutil
import sys


def print_usage():
    print("""
need args:
    <feature_dir_path> <data_dir_path>
    [do_clean]
where
    do_clean is <1|0>
""")


def main():
    if len(sys.argv) <= 2:
        print_usage()
        sys.exit(1)

    feature_dir_path, data_dir_path = sys.argv[1:3]
    do_clean = "0"
    if len(sys.argv) >= 3:
        do_clean = "1"

    if do_clean == "1" and os.path.exists(data_dir_path):
        print "cleaning %s" % data_dir_path
        shutil.rmtree(data_dir_path)
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)

    data_x_file_path = os.path.join(data_dir_path, "data_x.txt")
    data_y_file_path = os.path.join(data_dir_path, "data_y.txt")
    merge(feature_dir_path,
          data_x_file_path, data_y_file_path)
    if not post_check(data_x_file_path, data_y_file_path):
        return 1
    print "merged all into data_x_file: %s" % data_x_file_path
    print "merged all into data_y_file: %s" % data_y_file_path
    return 0


def merge(feature_dir_path, data_x_file_path, data_y_file_path):
    zero_line_cnt = 0
    with open(data_x_file_path, "w") as data_x_wfile, \
            open(data_y_file_path, "w") as data_y_wfile:
        for uid_fn in os.listdir(feature_dir_path):
            if not uid_fn.endswith(".uid"):
                continue
            uid = uid_fn.split(".uid")[0]
            uid_file_path = os.path.join(feature_dir_path, "%s.uid" % uid)
            label_file_path = os.path.join(feature_dir_path, "%s.label" % uid)
            with open(uid_file_path) as uid_rfile, \
                    open(label_file_path) as label_rfile:
                pics = uid_rfile.readlines()
                labels = label_rfile.readlines()
            if len(pics) != len(labels):
                raise StandardError("lines cnt cannot match of uid: %s" % uid)
            pic_cnt = len(pics)
            for line_no in xrange(pic_cnt):
                pic = pics[line_no].strip()
                label = labels[line_no].strip()
                if len(filter(lambda _: _ != "0", pic.split(","))) == 0:
                    zero_line_cnt += 1
                    continue
                if label not in ("0", "1"):
                    raise StandardError("illegal label %s in label_file %s" % (label, label_file_path))
                data_x_wfile.write(pic + "\n")
                if label == "1":
                    data_y_wfile.write("0,1" + "\n")
                elif label == "0":
                    data_y_wfile.write("1,0" + "\n")
    logging.info("zero_line_cnt: %s" % zero_line_cnt)


def post_check(data_x_file_path, data_y_file_path):
    with open(data_x_file_path, ) as data_x_rfile, \
            open(data_y_file_path, ) as data_y_rfile:
        if len(data_x_rfile.readlines()) != len(data_y_rfile.readlines()):
            logging.error("checking lens fail")
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
