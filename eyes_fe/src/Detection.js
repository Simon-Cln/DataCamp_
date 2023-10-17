import React, { useRef } from 'react';
import './Detection.css';
import 'aos/dist/aos.css';
import { useNavigate } from 'react-router-dom';
import EngineerGuy from './images/detectionimg.png';
/*import EngineerGuyScreenOn from './images/computer_on2.png';*/
import { useEffect, useState } from 'react';
/*import EngineerGuyScreenOn from './images/computer_ontest.png';*/
import EngineerGuyScreenOn from './images/yellowscreen2.png';


const Detection = () => {
    const inputRef = useRef();
    const navigate = useNavigate();

    const handleButtonClick = () => {
        inputRef.current.click();
    };

    const handleFileChange = event => {
        const file = event.target.files[0];
        if (file) {
            navigate('/loading', { state: { file } });
        } else {
            console.error("No file selected");
        }
    };
    const [scrolled, setScrolled] = useState(false);
    // eslint-disable-next-line
    const [imageSrc, setImageSrc] = useState(EngineerGuy);  // L'image par défaut est l'écran éteint


    useEffect(() => {
        const handleScroll = () => {
            setScrolled(true);
        };
    
        window.addEventListener('scroll', handleScroll);
    
        return () => {
            window.removeEventListener('scroll', handleScroll);
        };
    }, []);
    
    useEffect(() => {
        if (scrolled) {
            const timer = setTimeout(() => {
                const imageElementOn = document.querySelector(".image-guy");
                imageElementOn.classList.add("image-scan-effect");  // Ajoutez l'effet de balayage
    
                setTimeout(() => {
                    imageElementOn.style.opacity = "1";  // Rendez l'écran allumé visible
                    imageElementOn.classList.add("image-glow");
                    imageElementOn.classList.add("image-scanline");
                }, 1000);  // Attendez 1 seconde avant d'afficher l'écran allumé
    
            }, 1000);
        
            return () => {
                clearTimeout(timer);
            };
        }
    }, [scrolled]);
    
    
    
    return (
        <div className="detection-container">
            <div className="text-container" data-aos="fade-up">
                <div className="text-title">Detection</div>
                <div className="text-subtitle">Reveal the invisible with BreakThrough_: Eye disease detection in the age of AI...</div>
                <div className="text-description">
                    <strong>You're one click away from confirming what you think!</strong> <br/>
                    Just like a trusted friend, our AI technology is here to give you a second opinion. <br/><br/>
                    <strong>Upload your X-ray</strong> and let
                    <span style={{ color: 'black' }}> <strong>BreakThrough_</strong></span>'s advanced algorithms guide you to a clearer understanding. Because
                    we believe that technology should empower you, not complicate things. <br/>
                    <strong>So, go ahead, take that step, and embrace the certainty that BreakThrough brings.</strong>
                </div>

                <button className="play-button" onClick={handleButtonClick}>
                    Upload your Eye Image
                    <div className="vertical-half"></div>
                </button>
            </div>

            
            <img className="image image-scan-effect" src={EngineerGuy} alt="Guy with screen off" />
            <img className="image-guy" src={EngineerGuyScreenOn} alt="Guy with screen on" />
           

            <input 
                type="file"
                ref={inputRef}
                style={{display: "none"}}
                accept=".png"
                onChange={handleFileChange}
            />
        </div>
    );
};
export default Detection;