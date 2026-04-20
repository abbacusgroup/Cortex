class AbbacusCortex < Formula
  include Language::Python::Virtualenv

  desc "Cognitive knowledge system with formal ontology, reasoning, and intelligence serving"
  homepage "https://github.com/abbacusgroup/Cortex"
  url "https://files.pythonhosted.org/packages/source/a/abbacus-cortex/abbacus_cortex-0.3.3.tar.gz"
  sha256 "05e6bcd0ae5918a09eda1f005fd53cebd3834f64b83de63b2f9d48bf7dc39523"
  license "MIT"

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
      Run the setup wizard (first time only):
        cortex setup

      The wizard configures your LLM provider, installs embeddings,
      sets up background services, and registers with Claude Code.

      Or start manually:
        cortex serve --transport mcp-http
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
