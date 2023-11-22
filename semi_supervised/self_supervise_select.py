import sys
import os

# add parent directory to system path to be able to assess functions from root
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import os
import shutil
import argparse

# to do: 700-1000

parser = argparse.ArgumentParser(description="Extracts predicted grayscale images according to index. Have been selected manually in advance.")

# prefix
parser.add_argument("--flight_nr", default="flight_11", type=str, help="Number of flight.")
parser.add_argument("--date", default="220719_2", type=str, help="Date associated with flight. Contains identifier number if multiple flights were on that day.")

"""
FLIGHT 9

indeces_att_flight9 =  [80, 82, 86, 87, 90, 96, 103, 109, 114, 116, 117, 119, 120, 121, 123, 124, 128, 129, 144, 149, 
                        152, 153, 155, 156, 158, 199, 205, 206, 207, 208, 209, 231, 233, 245, 250, 260, 273, 305, 308, 
                        310, 656, 553, 554, 555, 556, 568, 579, 1005, 1006, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015]


indeces_att_flight916 = [412, 413, 417, 429, 431, 441, 442, 448, 449, 450, 453, 454, 457, 460, 461, 470, 472, 473, 480,
                          479, 481, 482, 483, 488, 489, 477, 501, 502, 505, 506, 507, 515, 516, 517, 523, 533, 538, 539, 
                          545, 547, 548, 549, 652, 653, 659, 661, 664, 665, 667, 668, 669, 670, 671, 672, 673, 674, 676, 
                          678, 679, 680, 682, 691, 692, 693, 694, 700, 510, 559, 560, 571, 572, 574, 597, 598, 599, 601, 
                          602, 605, 609, 610, 611, 614, 618, 627, 633, 637, 640, 649]

indeces_psp_flight916 = [683, 684, 689, 592, 594, 596]

#############################################################################################################################

indeces_flight7 = [103, 204, 205, 206, 207, 208, 209, 211, 212, 213, 214, 215, 217, 218, 228, 643, 622,
               621, 619, 618, 617, 603, 602, 600, 598, 595, 593, 567, 557, 537, 535, 378, 345, 298,
               289, 286, 253, 236, 230, 275, 276, 277, 680, 679, 649, 629, 650, 625, 616, 614, 605, 573]

indeces_flight16 = [1111, 1104, 1102, 1101, 1099, 1021, 1020, 1011, 723, 606, 595, 552, 540, 538, 531, 527, 316,
315, 312, 309]

no preselected indeces for flight 10

indeces_flight11 = [218, 213, 110, 111]

"""

def main():
    args = parser.parse_args()
    params = vars(args)

    indeces_att_flight9 = []
    indeces_att_flight916 = []
    indeces_psp_flight916 = []

    indeces = [218, 213, 110, 111]
               
    paths = []

    if params['flight_nr'] == 'flight_9':

        path_att_flight9 = os.path.join('data/prediction/predicted/att_unet_flight9/grayscale/')
        path_att_flight916 = os.path.join('data/prediction/predicted/att_unet/grayscale/')
        path_psp_flight916 = os.path.join('data/prediction/predicted/psp_net/grayscale/')
        paths.append(path_att_flight9, path_att_flight916, path_psp_flight916)

    else: 
        path = os.path.join('data/prediction/predicted/{}/grayscale/'.format(params['flight_nr']))
        paths.append(path)

    selected_path = os.path.join('data/selected/', params['date'])


    if params['flight_nr'] == 'flight_9':

        for idx in indeces_att_flight9:
            source = os.path.join(paths[0], '{}.png'.format(idx))
            dest = os.path.join(selected_path, '{}.png'.format(idx))
            shutil.copy(source, dest)

        for idx in indeces_att_flight916:
            source = os.path.join(paths[1], '{}.png'.format(idx))
            dest = os.path.join(selected_path, '{}.png'.format(idx))
            shutil.copy(source, dest)

        for idx in indeces_psp_flight916:
            source = os.path.join(paths[2], '{}.png'.format(idx))
            dest = os.path.join(selected_path, '{}.png'.format(idx))
            shutil.copy(source, dest)

    else:
        for idx in indeces:
            source = os.path.join(paths[0], '{}.png'.format(idx))
            dest = os.path.join(selected_path, '{}.png'.format(idx))
            shutil.copy(source, dest)

if __name__ == "__main__":
    main()