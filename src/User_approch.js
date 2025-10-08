import React, { useState } from "react";
import axios from "axios";
function User_approch() {
  const [yolov5xLower, setYolov5xLower] = useState("");
  const [yolov5xUpper, setYolov5xUpper] = useState("");
  const [yolov5lLower, setYolov5lLower] = useState("");
  const [yolov5lUpper, setYolov5lUpper] = useState("");
  const [yolov5mLower, setYolov5mLower] = useState("");
  const [yolov5mUpper, setYolov5mUpper] = useState("");
  const [yolov5sLower, setYolov5sLower] = useState("");
  const [yolov5sUpper, setYolov5sUpper] = useState("");
  const [yolov5nLower, setYolov5nLower] = useState("");
  const [yolov5nUpper, setYolov5nUpper] = useState("");
  const [finalized, setfinalized] = useState(false)
  const changeKnowledge= async () =>{
    try{
    const response = await axios.post('http://localhost:3001/apichangeKnowledge', {
        yolov5xLower,
        yolov5xUpper,
        yolov5lLower,
        yolov5lUpper,
        yolov5mLower,
        yolov5mUpper,
        yolov5sLower,
        yolov5sUpper,
        yolov5nLower,
        yolov5nUpper
      });
      alert("'Knowledge finalized")
      setfinalized(true)
      console.log(response.data); // Handle the response as needed

    } catch (error) {
      console.error(error);
    }
  };
  return (
    <div>
      {!finalized && <div>
      <div className="row">
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5xLower">
            Lower bound of response time for Xlarge model.
          </label>
          <input
            className="form-control"
            id="inputYolov5xLower"
            type="text"
            placeholder="Lower bound for yolov5x model."
            value={yolov5xLower}
            onChange={(e) => setYolov5xLower(e.target.value)}
          />
        </div>
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5xUpper">
          Upper bound of response time for Xlarge model
          </label>
          <input
            className="form-control"
            id="inputYolov5xUpper"
            type="text"
            placeholder="Upper bound for yolov5x model."
            value={yolov5xUpper}
            onChange={(e) => setYolov5xUpper(e.target.value)}
          />
        </div>
      </div>

      <div className="row mt-3">
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5lLower">
          Lower bound of response time for Large model
          </label>
          <input
            className="form-control"
            id="inputYolov5lLower"
            type="text"
            placeholder="Lower bound for yolov5l model."
            value={yolov5lLower}
            onChange={(e) => setYolov5lLower(e.target.value)}
          />
        </div>
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5lUpper">
          Upper bound of response time for Large model
          </label>
          <input
            className="form-control"
            id="inputYolov5lUpper"
            type="text"
            placeholder="Upper bound for yolov5l model."
            value={yolov5lUpper}
            onChange={(e) => setYolov5lUpper(e.target.value)}
          />
        </div>
      </div>
    
        
      <div className="row mt-3">
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5mLower">
          Lower bound of response time for Medium model
          </label>
          <input
            className="form-control"
            id="inputYolov5mLower"
            type="text"
            placeholder="Lower bound for yolov5m model."
            value={yolov5mLower}
            onChange={(e) => setYolov5mLower(e.target.value)}
          />
        </div>
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5mUpper">
          Upper bound of response time for Medium model
          </label>
          <input
            className="form-control"
            id="inputYolov5mUpper"
            type="text"
            placeholder="Upper bound for yolov5m model."
            value={yolov5mUpper}
            onChange={(e) => setYolov5mUpper(e.target.value)}
          />
        </div>
      </div>

      <div className="row mt-3">
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5sLower">
          Lower bound of response time for Small model
          </label>
          <input
            className="form-control"
            id="inputYolov5sLower"
            type="text"
            placeholder="Lower bound for yolov5s model."
            value={yolov5sLower}
            onChange={(e) => setYolov5sLower(e.target.value)}
          />
        </div>
        <div className="col-md-6">
          <label className="small" htmlFor="inputYolov5sUpper">
          Upper bound of response time for Small model
          </label>
          <input
            className="form-control"
            id="inputYolov5sUpper"
            type="text"
            placeholder="Upper bound for yolov5s model."
            value={yolov5sUpper}
            onChange={(e) => setYolov5sUpper(e.target.value)}
          />
        </div>

        <div className="row mt-3">
            <div className="col-md-6">
            <label className="small" htmlFor="inputYolov5nLower">
            Lower bound of response time for Nano model
            </label>
            <input
                className="form-control"
                id="inputYolov5nLower"
                type="text"
                placeholder="Lower bound for yolov5n model."
                value={yolov5nLower}
                onChange={(e) => setYolov5nLower(e.target.value)}
            />
            </div>
            <div className="col-md-6">
            <label className="small" htmlFor="inputYolov5nUpper">
            Upper bound of response time for Nano model
            </label>
            <input
                className="form-control"
                id="inputYolov5nUpper"
                type="text"
                placeholder="Upper bound for yolov5n model."
                value={yolov5nUpper}
                onChange={(e) => setYolov5nUpper(e.target.value)}
            />
            </div>
        </div>
      </div>


      <button className="btn mt-3 small btn-danger" onClick={changeKnowledge}>
          Final Knowledge
        </button>
        </div>}
    </div>
  );
}

export default User_approch;