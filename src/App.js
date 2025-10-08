import './App.css';
import Home from './Home';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import axios from 'axios';
import { useEffect } from 'react';

function App() {
  const startPythonScript = async () => {
    try {
      // Make a request to the server-side endpoint that executes the Python script
      await axios.post('http://localhost:3001/execute-python-script');
      console.log('Python script executed successfully');
    } catch (error) {
      console.error('Error executing Python script:', error);
    }
  };

  // Call the startPythonScript function when the component mounts
  useEffect(() => {
    startPythonScript();
  }, []);

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route exact path="/" element={<Home />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
