git clone https://github.com/tpn/pdfs.git ..
git clone https://github.com/gerpetrum/rupdfs.git ..

python3 ./script_generate_synthetic.py -s ../pdfs -r 300 -d ../Samples
python3 ./script_generate_synthetic.py -s ../rupdfs -r 300 -d ../Samples
