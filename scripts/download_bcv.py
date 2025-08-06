import os
import synapseclient
import synapseutils

# Get token from environment
syn_token = os.environ.get("SYNAPSE_TOKEN")
if not syn_token:
    raise RuntimeError("SYNAPSE_TOKEN environment variable not set.")

# Login
syn = synapseclient.Synapse()
syn.login(authToken=syn_token)

# Download BCV dataset using syncFromSynapse
files = synapseutils.syncFromSynapse(syn, 'syn3193805', path='src/data/bcv')
print("BCV dataset download complete.")
