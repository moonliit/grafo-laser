{
  description = "Frontend + backend runner";

  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs }: let
    pkgs = nixpkgs.pkgs;
    pythonEnv = pkgs.python312.withPackages (ps: with ps; [
      fastapi uvicorn paho-mqtt numpy matplotlib networkx
    ]);
  in {
    packages.default = pkgs.stdenv.mkDerivation {
      name = "laser-project";
      buildInputs = [ pythonEnv ];

      installPhase = ''
        mkdir -p $out/bin
        cat > $out/bin/start-all << 'EOF'
#!/usr/bin/env bash
set -e

echo "Starting frontend on port 5500..."
(cd frontend && python -m http.server 5500 &)

echo "Starting backend..."
(cd backend && python3 server.py)
EOF
        chmod +x $out/bin/start-all
      '';
    };

    apps.default = {
      type = "app";
      program = "${self.packages.default}/bin/start-all";
    };
  };
}
