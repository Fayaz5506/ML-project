import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat as nbf
import io
import contextlib
import base64

with open('Insurance_Claim_Fraud_Detection.ipynb', 'r') as f:
    nb = nbf.read(f, as_version=4)

global_env = {}

for idx, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        print(f"Executing code cell {idx + 1}...")
        code = cell.source
        stdout_capture = io.StringIO()
        plt.close('all')
        
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(code, global_env)
            
            output_str = stdout_capture.getvalue()
            outputs = []
            
            if output_str:
                outputs.append(nbf.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=output_str
                ))
            
            # Capture generated figures if any
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                outputs.append(nbf.v4.new_output(
                    output_type='display_data',
                    data={'image/png': img_b64, 'text/plain': '<Figure size ...>'}
                ))
                plt.close(fig)
                
            cell.outputs = outputs
            cell.execution_count = idx + 1
        except Exception as e:
            print(f"Error executing cell {idx + 1}: {e}")
            cell.outputs = [nbf.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=[str(e)]
            )]

with open('Insurance_Claim_Fraud_Detection.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook execution completed successfully with Agg backend!")
