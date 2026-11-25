import argparse
from .core import run

def main():
    p=argparse.ArgumentParser(description='Audit supplied label trajectories on fixed observed cells; no classification or inferential p-values.')
    p.add_argument('--cells', required=True, help='TSV: cell_id, region_id, x_um, y_um')
    p.add_argument('--labels', required=True, help='TSV: cell_id, replicate, label; baseline plus complete trajectories')
    p.add_argument('--pair', action='append', required=True, help='Two distinct labels separated by comma; repeat for more pairs')
    p.add_argument('--radius-um', type=float, default=20)
    p.add_argument('--out', required=True, help='New output directory; existing paths are refused')
    a=p.parse_args()
    try: run(a.cells,a.labels,a.pair,a.radius_um,a.out)
    except (ValueError,FileExistsError) as e:p.error(str(e))

if __name__=='__main__':main()
