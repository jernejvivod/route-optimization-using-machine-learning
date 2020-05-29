import glob
import os
import argparse

# Parse arguments.
parser = argparse.ArgumentParser(description='Repeat final frame')
grp = parser.add_mutually_exclusive_group()
parser.add_argument('n_rep', type=int, default=50, nargs='?', help='Number of times to repeat the last frame')
parser.add_argument('--undo', action='store_true', help='Undo last frame repetitions')
args = parser.parse_args()

# Get index of last frame in folder.
idx_last_frame = max(list(map(lambda x: int(x.split('.')[0]), map(os.path.basename, glob.glob('./animation_frames/*.png')))))

if args.undo:
    
    # If performing undo.
    with open('./animation_frames/.repetitions', 'a+') as f:

        # Parse number of frames to remove and update record.
        f.seek(0)
        n_rm = f.read()
        if n_rm != '':
            for rm_nxt in range(idx_last_frame, idx_last_frame - int(n_rm), -1):
                os.system('rm ./animation_frames/{0}.png'.format(rm_nxt))
            f.seek(0)
            f.truncate()
else:
    # If multiplying last frame.

    # Multiply last frame specified number of times.
    for n_nxt in range(idx_last_frame+1, idx_last_frame + 1 + args.n_rep):
        os.system('cp ./animation_frames/{0}.png ./animation_frames/{1}.png'.format(idx_last_frame, n_nxt))


    # Update multiplication count record.
    with open('./animation_frames/.repetitions', 'a+') as f:
        f.seek(0)
        data = f.read()
        if data != '':
            prev_rep = int(data)
        else:
            prev_rep = 0
        n_rep = args.n_rep + prev_rep
        f.seek(0)
        f.truncate()
        f.write(str(n_rep))


