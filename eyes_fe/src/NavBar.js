import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import { Link } from 'react-router-dom';
import './NavBar.css';
import logo from './images/logo_bt.png';
import { NavHashLink as NavLink } from 'react-router-hash-link';


const NavBar = () => {
    return (
        <Navbar bg="dark" expand="lg" sticky="top">
            <Navbar.Brand as={Link} to="/">
                <img 
                    alt="logo"
                    src={logo}
                    height="200"
                    className="d-inline-block align-top"
                />{' '}
            </Navbar.Brand>
            <Navbar.Collapse id="responsive-navbar-nav">
                <Nav className="ml-auto">
                <NavLink smooth to="#presentation" className="nav-link">
                    Presentation
                </NavLink>
                <NavLink smooth to="#detection" className="nav-link">
                    Detection
                </NavLink>
                </Nav>
            </Navbar.Collapse>
        </Navbar>
    );
}

export default NavBar;
