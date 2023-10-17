import { useNavigate, useLocation } from 'react-router-dom';
import React, { useState, useEffect } from 'react';
import './LoadingPage.css';
import axios from 'axios';

const LoadingPage = () => {
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState(null);
    const [prediction, setPrediction] = useState(null);
    // Utiliser le hook useLocation pour accéder à l'état du routeur
    const navigate = useNavigate();
    const location = useLocation();
    const loadingMessages = [
        "Analyzing Image...",
        "Detecting anomaly...",
        "Processing image...",
        "Running AI model...",
        "Finalizing results...",
        "Almost there...",
        "Loading...",
        "Please wait...",
    ];
    const [requestCompleted, setRequestCompleted] = useState(false);
    const [slideUp, setSlideUp] = useState(false);
    const [probability, setProbability] = useState(null);
    const [filename, setFilename] = useState(null);


    useEffect(() => {
        const file = location.state.file;
        console.log("Fichier recu du routeur : ", file);
        const formData = new FormData();
        formData.append("file", file);

    
        axios.post('http://localhost:5000/predict', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
            .then(response => {
                console.log("Réponse du serveur: ", response.data)
                setPrediction(response.data.prediction);
                setProbability(response.data.probability);
                setFilename(response.data.filename);
                console.log("Prédiction: ", response.data.prediction);
                setRequestCompleted(true);
            })
            .catch(err => {
                console.error("Erreur lors de la prédiction: ", err);
                if(err.response) {
                    console.error("Erreur de réponse: ", err.response.data);
                }
                setError(err.toString());
                setRequestCompleted(true);
            });
    }, [location.state.file]);

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress(oldProgress => {
                let newProgress = Math.min(oldProgress + 1, 100);
    
                // Si la progression atteint 100% et que la requête est terminée, naviguez vers la page de résultats
                if (newProgress === 100 && requestCompleted && !slideUp) {
                    clearInterval(interval);
                    setSlideUp(true);
                }
    
                return newProgress;
            });
        }, 60);
    
        return () => {
            clearInterval(interval);
        };
        // eslint-disable-next-line
    }, [requestCompleted, navigate, prediction, error]);
  
  
    useEffect(() => {
        if (slideUp) {
            const timer = setTimeout(() => {
                navigate('/results', { state: { prediction, error, probability,filename } });
            }, 1000);
            return () => clearTimeout(timer);
        }
    }, [slideUp, navigate, prediction, error, probability, filename]);


    // Utilisez une fonction pour obtenir un message aléatoire de la liste
    function getRandomLoadingMessage() {
        const randomIndex = Math.floor(Math.random() * loadingMessages.length);
        return loadingMessages[randomIndex];
    }

    // Ajoutez un état pour le message de chargement
    const [loadingMessage, setLoadingMessage] = useState(getRandomLoadingMessage());

    useEffect(() => {
        // Utilisez une fonction pour obtenir un message aléatoire de la liste
        function getRandomLoadingMessage() {
            const randomIndex = Math.floor(Math.random() * loadingMessages.length);
            return loadingMessages[randomIndex];
        }
        if (progress % 10 === 0) { // Change le message tous les 10%
            setLoadingMessage(getRandomLoadingMessage());
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [progress]);
    
    

    const color = `linear-gradient(to top, black ${progress}%, darkgray ${progress}%)`;


    return (
        <div className={`loading-container ${slideUp ? 'slide-up' : ''}`} style={{background: color}}>
            <div className="loading-container" style={{background: color}}>
                <div className="loading-bar" style={{ height: `${progress}%` }}></div>
                <div className="loading-text">{progress}%</div>
                <div className="loading-indicator">Chargement<span className="loading-dots">...</span></div>
                <div className="loading-message">{loadingMessage}</div>  {/* Affichez le message de chargement ici */}
            </div>
        </div>
    );
};
export default LoadingPage;