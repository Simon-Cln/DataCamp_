import React from 'react';
import { Link } from 'react-router-dom';
import './Footer.css';
import { FaFacebook, FaTwitter, FaInstagram } from 'react-icons/fa';

const Footer = () => {
    const year = new Date().getFullYear();  // Pour obtenir l'année courante
    return (
        <footer className="footer">
            <div className="social-media">
                <Link to="/"><FaFacebook /></Link>
                <Link to="/"><FaTwitter /></Link>
                <Link to="/"><FaInstagram /></Link>
            </div>
            <p>&copy; {year} - Simon Calarn</p>
        </footer>
    );
}

export default Footer;
