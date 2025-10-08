import React, { useState } from 'react';
import JSZip from 'jszip';
import Dashboard from './Dashboard';
import User_approch from './User_approch'
import User_approch_MAPE_K from './User_Approch_MAPE_K'
const Home = () => {
  const [selectedZipFile, setSelectedZipFile] = useState(null);
  const [selectedCSVFile, setSelectedCSVFile] = useState(null);
  const [showDashBoard ,  setShowDashBoard] = useState(false);
  const [stopProcessing , setstopProcessing] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');
  const [ID, setID] = useState('')
  const [loc, setLoc] = useState('')
  const handleZipFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedZipFile(file);
  };

  const handleCSVFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedCSVFile(file);
  };

  const handleLocationChange=(event)=>{
    setLoc(event.target.value);
    console.log(loc)
  }
  const handleIdChange = (event) => {
    // Update the ID state with the new value from the input field
    setID(event.target.value);
    console.log(ID)
  };

  const handleUpload = async () => {
    if ((!selectedZipFile && loc=='') || !selectedCSVFile) {
      return;
    }

    try {
      setShowDashBoard(true);
      
      if(selectedZipFile){
        const zip = new JSZip();
        await zip.loadAsync(selectedZipFile);

        const files = [];
        zip.forEach(async (relativePath, file) => {
          const content = await file.async('uint8array');
          files.push({ path: relativePath, content });
        });
      }
      

      const formData = new FormData();
      if (selectedZipFile) {
        formData.append('zipFile', selectedZipFile);
        console.log('Zip file added to foem data')
      }
      formData.append('csvFile', selectedCSVFile);
      formData.append('approch', selectedOption);
      if (loc) {
        formData.append('folder_location', loc);
      }
  
      console.log(selectedOption, loc)
      const response = await fetch('http://localhost:3001/api/upload', {
        method: 'POST',
        body: formData,
      });

      // Handle the response from the backend
      if (response.ok) {
        console.log('Files uploaded successfully.');
        setShowDashBoard(true);
        // var url = 'http://localhost:5601/app/dashboards#/list?_g=(filters:!(),refreshInterval:(pause:!t,value:0),time:(from:now%2Fd,to:now%2Fd))'
      //   if(window.confirm("View Dashboard")){
      //     window.open(url);
      //  }
        
      } else {
        console.error('Failed to upload files.');
      }
    } catch (error) {
      console.error('Error during file upload:', error);
    }
  };

  const stopProcess = async () => {
    try{
      const response = await fetch('http://localhost:3001/api/stopProcess', {
          method: 'POST'
        });
      if(response.ok){
        console.log("Stoped succesfully");
        setstopProcessing(true);
      }
      else{
        console.log("Failed to stop program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }
  };
  
  const newProcess = async ()=>{

    try{
      const response = await fetch('http://localhost:3001/api/newProcess', {
          method: 'POST'
        });
      if(response.ok){
        console.log("Restartes process");
        setstopProcessing(true);
        setID('')
        setSelectedCSVFile(null)
        setSelectedOption('')
        setLoc('')
        setSelectedZipFile(null)

      }
      else{
        console.log("Failed to restart program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }

    setShowDashBoard(false);
    setstopProcessing(false);
  }

  const downloadData= async ()=>{
    try{
      const response = await fetch('http://localhost:3001/api/downloadData', {
          method: 'POST',
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ data: ID }),
       
        });
      if(response.ok){
        console.log("Downloaded succesfully");
        alert("Downloaded Succesfully")
      }
      else{
        console.log("Failed to stop program")
      }
    }catch(error){
      console.error('Error during fstoping program:', error);
    }
  }

  const handleSelectChange = async(event) => {
    setSelectedOption(event.target.value);
    console.log(event.target.value)
    if(event.target.value === "NAIVE"){
      try{
        const response = await fetch('http://localhost:3001/useNaiveKnowledge', {
          method: 'POST'
        });
        if(response.ok){
          console.log("NAIVE knowledge updated");
          // alert("Downloaded Succesfully")
        }
        else{
          console.log("Failed load NAIVE knowledge")
        }
      }catch(error){
        console.error('Error during loading NAIVE knowledge:', error);
      }
    }
  };

  return (
    <div className="container mt-3 ">
      {!showDashBoard && <div>
        <h1 className="mb-3">SWITCH: An Exemplar for Evaluating Self-Adaptive ML-Enabled Systems</h1>
        <div className="mb-3">
          <label htmlFor="zipFileInput" className="form-label">
            Upload a .zip file, for folder contaning images, the .zip file must have same name as the Image folder.
          </label>
          <input
            type="file"
            className="form-control"
            id="zipFileInput"
            onChange={handleZipFileChange}
          />
        </div>

        <div className="mb-3">
          <label htmlFor="text" className="form-label">
            Upload folder base location if zip size greater than 700MB .
          </label>
          <input
            type="text"
            className="form-control"
            id="textInput"
            onChange={handleLocationChange}
          />
        </div>


        <div className="mb-3">
          <label htmlFor="csvFileInput" className="form-label">
            Upload a csv file contaning inter arrival rate data.
          </label>
          <input
            type="file"
            className="form-control"
            id="csvFileInput"
            onChange={handleCSVFileChange}
          />
        </div>
        <div>
        
        <div className="container">
      <div className="row justify-content-center">
        <div className="col-md-6">
          <div className="card">
            <div className="card-body">
              <h5 className="card-title">ID your Experiment</h5>
              <div className="form-group">
                <input
                  type="text"
                  id="IdInput"
                  className="form-control"
                  onChange={handleIdChange}
                  value={ID}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

        <div className="mb-3 mt-3 h5">
          <select className="selectpicker" value={selectedOption} onChange={handleSelectChange}>
            <option value="">Select an option</option>
            <option value="NAIVE">NAIVE</option>
            <option value="AdaMLs">AdaMLS</option>
            <option value="Try Your Own">Modify NAIVE</option>
            <option value="Write Your Own MAPE-K">Upload MAPE-K files</option>
            <option value="yolov5n">Nano Model</option>
            <option value="yolov5s">Small Model</option>
            <option value="yolov5m">Medium Model</option>
            <option value="yolov5l">Large Model</option>
            <option value="yolov5x">Xlarge Model</option>
          </select>
        </div>
        
        {selectedOption === "Try Your Own" &&

        <div className="mb-3">
            <User_approch/>
        </div>
        }

        {selectedOption === "Write Your Own MAPE-K" &&

        <div className="mb-3">
            <User_approch_MAPE_K
            id = {ID}/>
        </div>
        }

        </div>
        <button className="btn btn-primary " onClick={handleUpload}>
          Upload Files 
        </button>
      </div>}
      <div>
        {showDashBoard && 
        <div>
          <Dashboard/>
          {!stopProcessing &&
          <button className="btn btn-primary" onClick={stopProcess}>
            Stop Process
          </button>}
          {stopProcessing && 
          <p>
            <button className="btn btn-primary" onClick={downloadData}>
            Download Data
            </button>
          <br />
          <br />
            <button className="btn btn-primary" onClick={newProcess}>
            New Process
            </button> 
          </p>
          }
        </div>
        }
      </div>
    </div>
  );
};

export default Home;
