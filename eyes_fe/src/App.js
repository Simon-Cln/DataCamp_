import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import React, { useEffect } from 'react';
import 'aos/dist/aos.css';
import AOS from 'aos';
import './App.css';
import NavBar from './NavBar';
import Home from './Home';
import Verticaline  from './Verticaline';
import Footer from './Footer';
import LoadingPage from './LoadingPage';
import ResultsPage from './Results';

function App() {
  useEffect(() => {
    AOS.init({
      duration: 2500, 
    });
  }, []);

  return (
    <Router>
      <div className="App">
        <NavBar />
        <Verticaline />
        <Routes>
          <Route exact path="/" element={<Home />} />
          <Route path="/loading" element={<LoadingPage />} />
          <Route path="/results" element={<ResultsPage />} />
        </Routes>
        <Footer />
      </div>
    </Router>
  );
}

export default App;
