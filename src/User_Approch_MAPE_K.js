import React, { useState } from 'react';
import axios from 'axios';

function FileUpload({id}) {
  const [files, setFiles] = useState({
    monitor: null,
    analyzer: null,
    planner: null,
    execute: null,
    knowledge: null,
  });
  const [finalized, setfinalized] = useState(false)
  const handleFileChange = (event, type) => {
    setFiles({
      ...files,
      [type]: event.target.files[0],
    });
  };

  const handleUpload = async () => {
    const formData = new FormData();

    // Add the selected files to the FormData object
    formData.append('monitor', files.monitor);
    formData.append('analyzer', files.analyzer);
    formData.append('planner', files.planner);
    formData.append('execute', files.execute);
    formData.append('knowledge', files.knowledge);
    formData.append('id', id);
    // Send the FormData to the FastAPI backend
    try {
      console.log(id)
      await axios.post('http://localhost:3001/your_mape_k', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setfinalized(true)
      alert('MAPE-K Files uploaded successfully.');
    } catch (error) {
      console.error('Error uploading files:', error);
    }
  };

  return (
    <div>
    {!finalized && <div>
      <h2>Upload files for MAPE-K</h2>
      <table className="table">
        <tbody>
          <tr>
            <td>
              <label htmlFor="monitor" className="form-label">Monitor.py</label>
              <input
                type="file"
                accept=".py"
                className="form-control"
                id="monitor"
                onChange={(e) => handleFileChange(e, 'monitor')}
              />
            </td>
            <td>
              <label htmlFor="analyzer" className="form-label">Analyzer.py</label>
              <input
                type="file"
                accept=".py"
                className="form-control"
                id="analyzer"
                onChange={(e) => handleFileChange(e, 'analyzer')}
              />
            </td>
          </tr>
          <tr>
            <td>
              <label htmlFor="planner" className="form-label">Planner.py</label>
              <input
                type="file"
                accept=".py"
                className="form-control"
                id="planner"
                onChange={(e) => handleFileChange(e, 'planner')}
              />
            </td>
            <td>
              <label htmlFor="execute" className="form-label">Execute.py</label>
              <input
                type="file"
                accept=".py"
                className="form-control"
                id="execute"
                onChange={(e) => handleFileChange(e, 'execute')}
              />
            </td>
          </tr>
        </tbody>
      </table>

      <div className="mb-3">
        <label htmlFor="knowledge" className="form-label">Knowledge (Zip File)</label>
        <input
          type="file"
          accept=".zip"
          className="form-control"
          id="knowledge"
          onChange={(e) => handleFileChange(e, 'knowledge')}
        />
      </div>

      <button onClick={handleUpload} className="btn btn-danger">Upload files for MAPE-K</button>
    </div>}
    </div>
  );
}

export default FileUpload;
