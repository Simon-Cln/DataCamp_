import React from 'react';
import './Presentation.css';
import 'aos/dist/aos.css';

import sampleVideo from './video/ltvid_pres.mp4';

const Presentation = () => {
    return (
        <div className="presentation-container">
            
            <div className="text-containers" data-aos="fade-up">
                <div className="text-title">Presentation</div>
                <div className="text-subtitle">BreakThrough_: Detect, Heal, Repeat.</div>
                <div className="intro-texts">
                    BreakThrough_ is a medical imaging Project that uses artificial intelligence to detect eye abnormalities and potential cancers in eye images. Our mission is to provide a fast, accurate and affordable solution for eye disease detection through image analysis.
                </div>
                <div className="video-text-container">
                    <div className="video-container">
                        <video className="video" src={sampleVideo} controls />
                    </div>
                    <div className="text-descriptions">
                        Defy uncertainty and take control. Whether you're a healthcare professional or someone looking for answers, our technology is here to serve you. The power of AI, coupled with your intuition, can help illuminate what may be hidden to the naked eye. Your decision to act today can be the first step towards a revolution in ocular disease detection. Let us help you unveil the invisible and bring certainty where it's needed. Upload your eye image and let BreakThrough do the rest. The future of eye disease detection is at hand!
                    </div>
                </div>
            </div>
        </div>
    );
}

export default Presentation;

