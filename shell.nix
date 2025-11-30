{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python312;                      # choose python version
  pythonEnv = python.withPackages (ps: with ps; [
    pip
    numpy
    matplotlib
    networkx
    fastapi
    uvicorn
    paho-mqtt
  ]);
in

pkgs.mkShell {
  name = "python-graph-env";

  buildInputs = [
    pythonEnv    # Python with the packages we need
  ];

  # small convenience: avoid picking up user site packages
  shellHook = ''
    export PYTHONNOUSERSITE=1
    printf "Entered nix-shell: python=%s\n" "${python.interpreter}"
  '';
}
