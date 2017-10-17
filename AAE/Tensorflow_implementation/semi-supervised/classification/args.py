import argparse

'''

wrap the code in order to make the interface user-friendly.

'''
parser = argparse.ArgumentParser(description='adversarial autoencoder for minist')
parser.add_argument('-p','--log_path',type=str,help='the path of saving your model')
#
print_style = parser.add_mutually_exclusive_group()
print_style.add_argument('-t','--table',action='store_true',help='the table like style')
print_style.add_argument('-b','--process_bar',action='store_true',help='style with process bar')

args = parser.parse_args()