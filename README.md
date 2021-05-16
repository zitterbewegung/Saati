satti
==============================

Saati is an open source virtual dating / idol simulator that uses generative models 

The project is inspired by the video games like AI Dungeon, visual novels and virtual youtubers.

The main concept is to make an open ended video game that primarily uses language models and sentiment analysis to advance story progression and also provide visual stimuli that goes up to the limit of the uncanny valley.

The success of virtual youtubers Calliope and Kiara who are a part of hololive-EN show that this kind of interface can avoid this until computer vision techniques can get us past that. 

How this works is that once you interact by saying something to the virtual agent it will query the language model with the response and also perform sentiment analysis on your interaction. Then we have a running total of positive and negative things that you say to the virtual agent. If the amount of positive and negative interactions is five to one then you will advance in the simulation / game. Advancement will eventually unlock new ways to contact the system. In other words advancement is done by achievement and other ways to perform sentiment analysis may become in play. 

This could also be applied using a therapeutic way of lowering your anxiety by making the interactions in dating let you desensitize yourself with the interactions.

To accomplish this I use the transformers library to perform sentiment analysis on an incoming message. After performing sentiment analysis I compute what I call is a synchronization ratio  is computed which corresponds to the ratio of the positive and negative interactions. What is also kept track of is the total amount of interactions. I give eleven interactions as a mulligan value to allow for the five to one ratio. If the mulligan value is exceeded and the sync_ratio isn’t fulfilled then the game will put you into the friend zone state and you can’t get out of the friend zone state.

The actual gameplay currently uses blenderbot which is a model created by Facebook. I evaluated it against dialogpt from Microsoft and in my testing blenderbot was a much better system.  I also use a distilled version with one tenth of the parameters so that the interactions are perceived in realtime. Using larger versions of blenderbot with my current hardware makes responses take over one minute.

I record state currently by using an identifier (for web chat it uses an IP address and for sms it is the incoming phone number).

I use Colab Pro to prototype new features and eventually encapsulate it into a function and then I put it in the inference_functions in the src directory. 

A modular and extendable visual vocal assistant. The modules that are planned of highest priority:

1. A socialization simulator
2. Vocal assistant
3. Push and pull behavior to provide general reminders
4. A system to encourage good behavior

Development log below: 

https://youtu.be/244lvaXevEE

This is licenced under the AGPL.
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![AGPL License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="https://raw.githubusercontent.com/zitterbewegung/Saati/master/assets/images/logo.png" alt="Logo">
  </a>

  <h3 align="center">project_title</h3>

  <p align="center">
    project_description
    <br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS 
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT 
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`, `project_title`, `project_description`


### Built With

* []()
* []()
* []()



<!-- GETTING STARTED 
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```



<!-- USAGE EXAMPLES
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the AGPL License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/zitterbewegung) - email

Project Link: [https://github.com/zitterbewegung/Saati](https://github.com/zitterbewegung/Saati)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
