%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This is a class for LaTeX2e that should format a report as per
% the IITM Thesis guidelines.  The page layout is set using the
% geometry package.  The double/singlespacing is setup using
% setspace.  Figure captions are set using the hang option of
% caption2.  natbib is used to do the citation using an
% author-year format. I've made an iitm.bst file that formats 
% the references as per the IITM guidelines.
%
% To use this class do the following:
%  \documentclass[options]{nitkdiss} 
%
% Options:
%
%   PhD, MS, MTech or BTech -- generates the appropriate title page
%   and loads the report class.
%
%   synopsis -- Generates the title page for the synopsis.  This also
%   loads the article class instead of the report.
%  
% Example:
%
% \documentclass[PhD,synopsis]{nitkdiss}
% \documentclass[MS]{nitkdiss}
%
% IMPORTANT NOTICE:
% 
%   PLEASE DO NOT MESS WITH THE MARGINS AND GENERAL FORMATTING SETUP
%   IN THIS FILE UNLESS YOU ARE ABSOLUTELY SURE THAT THE FORMAT DOES NOT
%   MATCH WITH THE THESIS GUIDELINES.  FOR EXAMPLE, DO NOT CHANGE THE 
%   MARGINS AND SPACING JUST TO MAKE YOUR THESIS LARGER!
% 
% Notes:
%
%   * I am using as much of the Thesis guidelines for the spacing
%     and margins as I can.
%   * I have used newdiss.cls by R.~K.~Hariram, U.~V.~Ravindra et al. 
%     as a reference and a source for some of the macros.
%   * This class will assume a4paper with 12pt fonts.
%   * I am no TeXpert so feel free to clean up the mess.
%
% Prabhu Ramachandran <prabhu@ae.iitm.ac.in>
% March 2005.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{nitkdiss}[2005/03/09 Class for NITK Dissertations]

\newif\if@PhD
\newif\if@MS
\newif\if@MTech
\newif\if@BTech
\newif\if@Synopsis

\DeclareOption{PhD}{\@PhDtrue}
\DeclareOption{MS}{\@MStrue}
\DeclareOption{MTech}{\@MTechtrue}
\DeclareOption{BTech}{\@BTechtrue}
\DeclareOption{synopsis}{\@Synopsistrue}
\ProcessOptions\relax

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Load the correct class.
\if@Synopsis
    \LoadClass[12pt,a4paper]{article}
\else
    \LoadClass[12pt,a4paper]{report}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geometry and page layout
\RequirePackage[a4paper,left=1in,right=1in,top=1in,bottom=0.667in, nohead]{geometry}[2002/07/08]
\parskip 8pt

\renewcommand{\@pnumwidth}{2.5em}
%\renewcommand{\@tocrmarg}{2.5em}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Using setspace to do the single/double spacing.  Setspace also
%% makes all tables and figures have single spacing automatically.
\RequirePackage{setspace} 
\RequirePackage{atbeginend} % used to redefine the list environs
\setstretch{1.5}

\BeforeBegin{itemize}{\vspace*{-5ex}\begin{singlespace}}
\AfterEnd{itemize}{\end{singlespace}}

\BeforeBegin{enumerate}{\vspace*{-5ex}\begin{singlespace}}
\AfterEnd{enumerate}{\end{singlespace}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The figure captioning is done correctly via the hang option.
\RequirePackage[hang]{caption}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bibliography
% 
% natbib allows easy switching of citation style and using it along
% with the iitm.bst makes things perfect.
%\RequirePackage[authoryear,round]{natbib} 
%\bibliographystyle{nitk}
%\newcommand{\bibstyle@nitk}{\bibpunct{(}{)}{;}{a}{,}{,}%
%     \gdef\NAT@biblabel##1{##1. }}

%\setlength{\bibhang}{0pt}
%\citestyle{iitm}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Other useful macros.  Many of these are modifications from newdiss.cls
% \registernumber{register number of the student} -- user macro to declare name of department.
\def\registernumber#1{\gdef\@registernumber{#1}}
% 
% \department{department name} -- user macro to declare name of department.
\def\department#1{\gdef\@department{#1}}

\def\mentor#1{\gdef\@mentor{#1}}

\RequirePackage[dvips]{graphicx}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The title page
\def\maketitle{%
  \pagenumbering{roman}
  \thispagestyle{empty}
  \newpage
  \null
  \setcounter{page}{0}
  \parskip=0pt
  \begin{center}%
  \let \footnote \thanks
  \vglue 0in % this makes top margin 2in
  \vskip -5ex%
  \if@Synopsis
    {\large SYNOPSIS OF \par}
  \fi
    {\LARGE \bf \expandafter{\@title} \par}%
    \vskip 10ex%
    {\normalsize
      \if@PhD
         {\large Thesis }\par
%          \vskip 4ex%
         \if@Synopsis
            {\normalsize \em to be submitted by \par}
         \else
            {\large Submitted in partial fulfillment of the requirements for the degree of}  \par
         \fi
	\vskip 3ex%
	 {\bf \Large \MakeUppercase{Doctor of Philosophy}} \par
         \vskip 3ex%
	{\large by \par}
         \vskip 3ex%
         {\bf \Large\MakeUppercase{\@author}\par\par}
%          \vskip 5ex%
%          {\em for the award of the degree }\par
%          \vskip 5ex%
%          {\em of} \par
%          \vskip 5ex%
%          
       \else
         \if@MS
            {\em A THESIS }\par
            \vskip 5ex%
            {\normalsize \em submitted by \par}
            \vskip 5ex%
            {\bf \large\MakeUppercase{\@author}\par\par}
            \vskip 5ex%
            {\em for the award of the degree }\par
            \vskip 5ex%
            {\em of} \par
            \vskip 5ex%
            {\bf \large \MakeUppercase{Master Of Science}} \par
            (by Research) \par
         \else
           {\em A Project Report }\par
           %CHANGE
	   %\vskip 5ex%
           {\normalsize \em submitted by \par}
           %\vskip 5ex%
           %CHANGE
	   %{\bf \large\MakeUppercase{\@author}\par\par}
	   {\bf \large{\@author}\par\par}
           \vskip 5ex%
           {\normalsize \em under the guidance of \par}
           {\bf \large{\@mentor}\par\par}
           \vskip 5ex%
           {\normalsize \em in partial fulfilment of the requirements\par
             for the award of the degree of \par}
           %\vskip 5ex%
           \if@MTech
             {\bf MASTER OF TECHNOLOGY}
           \fi
           \if@BTech
              {\bf BACHELOR OF TECHNOLOGY}
           \fi
         \fi
      \fi
    }
  \end{center}%

%CHANGE
 % \vspace*{0.75in}
  %\centerline{\psfig{figure=emblem.ps}}
%CHANGE - nitk.eps to nitk.jpg
  \centerline{\includegraphics[width=2in]{nitk}}
%CHANGE 
 %\vspace*{0.25in}
  \begin{center}
%CHANGE
    %{\bf \large DEPARTMENT OF \MakeUppercase{\@department}\par}
    %{\bf \large NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA}
    %{\bf \large SURATHKAL, MANGALORE - 575025}
    {\bf  DEPARTMENT OF \MakeUppercase{\@department}\par}
    {\bf  NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA}
    {\bf  SURATHKAL, MANGALORE - 575025}
    \vglue 0.50em
    {\bf \large \@date}\par
  \end{center}
  \parskip 8pt
}

\def\titlepage{\newpage\thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\newpage}

\if@Synopsis
   \renewcommand\refname{REFERENCES}
\else
   % QUOTATIONS
   %
   \def\quotations{\chapter*{\centerline{QUOTATIONS}}
   %\addcontentsline{toc}{chapter}{QUOTATIONS}
   }

   % DEDICATION
   %
   \def\dedication{\chapter*{\centerline{DEDICATION}}
   %\addcontentsline{toc}{chapter}{DEDICATION}
   }

   % DECLARATIONS
   %
   %   - this produces a declaration page as per NITK Regulations
   %     comment the following lines out to get it working.
   %
   \def\declaration{%% A preliminary chapter
     \clearpage
     \vspace*{36pt}
     \centerline{\Large \bf DECLARATION}
     \centerline{\bf \textit{by the B.Tech students}}
     \vskip 25pt
     \thispagestyle{empty}
%      \typeout{Certificate}
\vspace*{0.5in}

\noindent We hereby {\bf\textit{declare}} that the Project entitled {\bf \@title} which is being submitted to the {\bf\textit{National Institute of Technology Karnataka, Surathkal}} in partial fulfillment of the requirements for the award of the Degree of {\bf\textit{Bachelor of Technology}} is a {\bf\textit{bonafide report of the research work carried out by us}}. The material contained in this thesis has not been submitted to any University or Institution for the award of any degree.

\vspace*{1in}
\begin{singlespacing}
\hspace*{2.5in} 
\parbox{2.5in}{
\noindent {\bf \@author}\\
\noindent {Department of \@department}\\
}
\end{singlespacing} 
% 
\vspace*{0.5cm}

\noindent Place: NITK - Surathkal\\
\noindent Date: \@date
     \setcounter{page}{0}
   }

 %%% CERTIFICATE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %
   %   - this produces a certificate page as per IITM Regulations
   %     comment the following lines out to get it working.
   %
   \def\certificate{%% A preliminary chapter
     \clearpage
     \vspace*{36pt}
     \centerline{\Large \bf CERTIFICATE}
     \vskip 25pt
     \thispagestyle{empty}
     \typeout{Certificate}
% \vspace*{0.25in}
     \setcounter{page}{0}
   }
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % ACKNOWLEDGEMENTS
   %
   \def\acknowledgements{\chapter*{\centerline{ACKNOWLEDGEMENTS}}
     \addcontentsline{toc}{chapter}{ACKNOWLEDGEMENTS}
   }

   % ABSTRACT
   %   - this produces pages suitable for an IITM Abstract
   \def\abstract{\chapter*{\centerline{ABSTRACT}}
   \addcontentsline{toc}{chapter}{ABSTRACT}
   }

   % GLOSSARY
   %
   \def\glossary{\chapter*{\centerline{GLOSSARY}}
   \addcontentsline{toc}{chapter}{GLOSSARY}
   }

   % ABBREVIATIONS
   %
   \def\abbreviations{\chapter*{\centerline{ABBREVIATIONS}}
   \addcontentsline{toc}{chapter}{ABBREVIATIONS}
   }

   % Use the nomencl package for the NOTATION section.

   % Makes the chapter heading
   \def\@makechapterhead#1{%
     {\parindent \z@ \raggedright \normalfont
       \ifnum \c@secnumdepth >\m@ne
       \begin{center}
         \Large\bfseries \@chapapp\space \thechapter
       \end{center}
       %\par\nobreak
       %\vskip 5\p@
       \vspace{-0.5in}
       \fi
       %\interlinepenalty\@M
       \Large \bf
       \begin{center}
         #1\par\nobreak      
       \end{center}
       %\vskip 5\p@
       \thispagestyle{empty}
     }}

   % Makes the chapter heading for \chapter* commands
   \def\@makeschapterhead#1{%
     \vspace*{25\p@}%
     {\parindent \z@ \raggedright
       \normalfont
       \interlinepenalty\@M
       \Large \bfseries #1\par\nobreak
       \vskip 20\p@
     }}


   %LIST OF PAPERS
   \def\listofpapers{
           \normalbaselines
           \chapter*{\centerline{LIST OF PAPERS BASED ON THESIS}}
           \markboth{LIST OF PAPERS BASED ON THESIS}{LIST OF PAPERS BASED ON THESIS}
           %\addcontentsline{toc}{chapter}{LIST OF PAPERS BASED ON THESIS}
   }

   
   % CV
   \def\CV{\chapter*{\centerline{CURRICULUM VITAE}}}

   % Comittee
   \if@PhD
       \def\committee{\chapter*{\centerline{DOCTORAL COMMITTEE}}}
   \else
       \if@MS
          \def\committee{\chapter*{\centerline{GENERAL TEST COMITTEE}}}}
        \fi
   \fi


   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Renewed commands to set the titles of various pages correctly.
   \renewcommand\contentsname{\centering TABLE OF CONTENTS}
   \renewcommand\listfigurename{\centering LIST OF FIGURES}
   \renewcommand\listtablename{\centering LIST OF TABLES}
   \renewcommand{\chaptername}{CHAPTER}
   \renewcommand\bibname{\centering REFERENCES}
   \renewcommand\appendixname{APPENDIX}
\fi


