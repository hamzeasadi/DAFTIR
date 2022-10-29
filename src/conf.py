import os



# define paths
root = os.pardir
data_root = os.path.join(root, 'data')
paths = dict(
    data=data_root, clean_data=os.path.join(data_root, 'clean_data'), 

    m5_spec=os.path.join(data_root, 'clean_data', 'm5_spec.csv'), 
    mp5_spec=os.path.join(data_root, 'clean_data', 'mp5_spec.csv'),
    mp6_spec=os.path.join(data_root, 'clean_data', 'mp6_spec.csv'),

    label0=os.path.join(data_root, 'clean_data', 'label0.csv'),
    label1=os.path.join(data_root, 'clean_data', 'label1.csv'),
    label2=os.path.join(data_root, 'clean_data', 'label2.csv'), 
    label3=os.path.join(data_root, 'clean_data', 'label3.csv'),

    ckpoint=os.path.join(data_root, 'ckpoint')
)





# model template
model_temp = dict(
    blk1=dict(inch=2, outch=8, ks=9, stride=5, pool=False, dropout=False),
    blk2=dict(inch=8, outch=8, ks=4, stride=2, pool=False, dropout=True),
    blk3=dict(inch=8, outch=8, ks=4, stride=2, pool=False, dropout=True),
    blk4=dict(inch=8, outch=8, ks=3, stride=1, pool=True, dropout=True),

)

def conv_shape(layer: dict, input_size: int):
    out_size = int((input_size - layer['ks'])/layer['stride']) + 1
    return out_size


def main():
    # key = list(paths.keys())
    # print(key)
    in_size = 140
    for key in model_temp.keys():
        out_size = conv_shape(layer=model_temp[key], input_size=in_size)
        in_size = out_size
        print(f"out-size={in_size}")



if __name__ == '__main__':
    main()