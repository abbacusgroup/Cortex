class AbbacusCortex < Formula
  include Language::Python::Virtualenv

  desc "AI knowledge pipeline -- capture, search, and reason over your knowledge"
  homepage "https://github.com/abbacusgroup/Cortex"
  url "https://files.pythonhosted.org/packages/source/a/abbacus-cortex/abbacus_cortex-0.2.2.tar.gz"
  sha256 "88f3ce14c64e3c5826fa2ee3ee273f2b6216b737835a99d3d53146106444dc5a"
  license "BUSL-1.1"

  depends_on "python@3.12"

  def install
    python3 = "python3.12"
    venv = virtualenv_create(libexec, python3)
    # Install from PyPI wheel (avoids building from source)
    system libexec/"bin/python", "-m", "ensurepip", "--default-pip"
    system libexec/"bin/python", "-m", "pip", "install", "--upgrade", "pip"
    system libexec/"bin/python", "-m", "pip", "install", "abbacus-cortex==#{version}"
    bin.install_symlink libexec/"bin/cortex"
  end

  def post_install
    (var/"cortex").mkpath
    (var/"log/cortex").mkpath
  end

  def caveats
    <<~EOS
      To start the Cortex MCP server as a background service:
        brew services start abbacus-cortex

      Or run manually:
        cortex serve --transport mcp-http

      Initialize Cortex (first time only):
        cortex init

      For semantic search, install the embeddings extra:
        #{libexec}/bin/pip install sentence-transformers

      Or configure an API-based embedding provider:
        export CORTEX_EMBEDDING_PROVIDER=litellm
        export CORTEX_EMBEDDING_MODEL=openai/text-embedding-3-small
        export CORTEX_EMBEDDING_API_KEY=sk-...
    EOS
  end

  service do
    run [opt_bin/"cortex", "serve", "--transport", "mcp-http",
         "--host", "127.0.0.1", "--port", "1314"]
    keep_alive true
    log_path var/"log/cortex/mcp.log"
    error_log_path var/"log/cortex/mcp.err"
    working_dir var/"cortex"
  end

  test do
    system bin/"cortex", "--help"
  end
end
