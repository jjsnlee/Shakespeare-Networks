import sys
from shakespeare.plays_transform import generate_shakespeare_files


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description='Prepare data files for web page usage.')

    parser.add_argument('whichCorpus', type=str, 
                        help='[shakespeare|gutenberg] (default: shakespeare)',
                        default='shakespeare')
    
    parser.add_argument('--gen_json', help='Generate backend JSON files', action='store_true')
    parser.add_argument('--gen_imgs', help='Generate backend image files', action='store_true')
    args = parser.parse_args()
    # print args

    if args.whichCorpus=='shakespeare':
        gen_imgs = gen_md = gen_lines = False

        if args.gen_imgs:
            gen_imgs = True
        if args.gen_json:
            gen_md = gen_lines = True

        print(gen_imgs, gen_md, gen_lines)
        
        generate_shakespeare_files(gen_imgs=gen_imgs, gen_md=gen_md, gen_lines=gen_lines)
    
#    elif args[0]=='gutenberg':
#        main_gutenberg()


if __name__ == "__main__":
    main(sys.argv)
