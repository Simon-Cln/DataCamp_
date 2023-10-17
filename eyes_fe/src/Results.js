import React, { useState } from 'react';  // Ajout de useState
import './Results.css';
import { useLocation, useNavigate } from 'react-router-dom'; 
import { useEffect } from 'react';

const Results = () => {
    const location = useLocation();
    console.log("Données recues de LoadingPage.js: ", location.state);
    const navigate = useNavigate();
    const prediction = location.state?.prediction;
    const error = location.state?.error;
    const probability = location.state?.probability;
    const filename = location.state?.filename;
    const [showText, setShowText] = useState(false);



    useEffect(() => {
        // Désactiver le défilement lorsque le composant est monté
        document.body.style.overflowY = 'hidden';
    
        // Réactiver le défilement lorsque le composant est démonté
        return () => {
            document.body.style.overflowY = 'auto';
        };
    }, []);
    

    return (
        <div className="results-container">
            <div className="box box-1"></div>
            <div className="box2 box-1"></div>

            {/* Ajout de l'image 3D */}
            <div className="doctor-image" onClick={() => navigate('/')}>
                <img src="/resultdocssss.png" alt="3D Doctor" />
            </div>


            <div className="prediction-result">
                <h1>Diagnosis</h1>
                <p>Your uploaded image has been processed and analyzed by our advanced neural network model. <span className="underline-text">Below are the findings</span></p>
            </div>


            {prediction && (
                <div className="text-below-image">
                    <p>This diagnosis is based on the CNN model trained on 2000 eye images.</p>
                </div>
            )}

            {prediction === "NORMAL" && (
                <div className="model-info">
                    <h3>Analysis</h3>
                    <p>Based on our <span className="highlight">analysis</span>, it seems your eye image <span className="important">does not indicate an anomaly</span> or disease. Nonetheless, if you're experiencing <span className="warning">vision issues, discomfort, or any unusual symptoms</span>, it's still imperative to <span className="advice">consult an ophthalmologist</span>. Some eye conditions might not be detectable through an image analysis and might require a thorough examination.</p>
                </div>
            )}


            {prediction === "ANOMALY" && (
                <div className="model-info">
                    <h3>Advice</h3>
                    <p>Based on our <span className="highlight">analysis</span>, it seems that there might be a <span className="warning">disease.</span> Eye diseases can vary significantly in terms of severity and type. It's <span className="important">essential to consult a doctor immediately for a comprehensive eye examination</span>. Proper medical attention is crucial to ensure correct healing and prevent potential vision loss or other complications.</p>
                    <p>It's <span className="advice">recommended to consult a doctor immediately</span></p>
                </div>
            )}

            {filename && (
                <div className="image-containerrr">
                    <img src={`http://localhost:5000/uploads/${filename}`} alt="Uploaded prediction" />
                </div>
            )}


<div className="analysis-container">
                
                {probability && (
                        <div className="bottom-left-text">
                            <div className="certainty-bar" 
                                onMouseEnter={() => setShowText(true)}
                                onMouseLeave={() => setShowText(false)}>
                                <div className="certainty-bar-fill" style={{width: `${probability * 100}%`}}></div>
                                {!showText && <p className="hover-prompt">Hover over the bar to reveal probability</p>}
                            </div>
                        </div>
                    )}

                    {showText && (
                        <div className="hover-text-container">
                            <p>{Math.round(probability * 100)}% sure</p>
                            <p>Please consult a healthcare professional for a final diagnosis. The model's predictions are based on patterns and may not be definitive.</p>
                        </div>
                    )}



                {error && (
                    <div className="error-message">
                        <h3>Erreur:</h3>
                        <p>{error}</p>
                    </div>
                )}
            </div>  {/* Cette fermeture de div manquait */}
            <div className="arrow"></div>
        </div>
    );
};

export default Results;

/*<h3>Certainty:</h3>*/
