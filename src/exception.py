# ============================================================
# AeroGuard — Custom Exception Handler
# 
# Yeh file poore project ki custom exception handling karti hai
#
# Problem jo yeh solve karta hai:
# Python ka default error message aisa hota hai:
#   "FileNotFoundError: file.csv not found"
# Isse pata nahi chalta — kaunsi file, kahan, kyun, kab
#
# Hamara custom exception aisa dikhega:
#   "AeroGuardException: [flight_header.csv] not found
#    Location: src/data/ingestion.py, Line 45
#    Context: Loading NGAFID flight header data"
#
# Usage:
#   from src.exception import AeroGuardException
#   raise AeroGuardException("Kuch galat hua", context="Data loading")
# ============================================================

import sys
import traceback
from src.logger import logger


def get_error_details(error: Exception) -> dict:
    """
    Kisi bhi exception se poori detail extract karta hai.
    
    Yeh function Python ke sys module se error ki exact location
    nikalta hai — kaunsi file, kaunsi line number, kya message.
    
    Args:
        error: Jo bhi exception aaya ho
        
    Returns:
        dict with file_name, line_number, error_message
    """
    # sys.exc_info() teen cheezein deta hai:
    # [0] = error type (jaise FileNotFoundError)
    # [1] = error object
    # [2] = traceback object — yahan exact location hoti hai
    exc_type, exc_value, exc_tb = sys.exc_info()

    # Agar traceback available hai toh location nikalo
    if exc_tb is not None:
        # tb_frame = current stack frame
        # f_code = code object
        # co_filename = file ka path
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = 0

    return {
        "file_name": file_name,
        "line_number": line_number,
        "error_message": str(error),
        "error_type": type(error).__name__
    }


class AeroGuardException(Exception):
    """
    AeroGuard ka master custom exception class.
    
    Poore project mein jahan bhi koi error aaye, yeh class
    use hogi taaki consistent aur detailed error messages milein.
    
    Yeh Python ke built-in Exception class ko inherit karta hai —
    matlab yeh ek proper Exception hai jo try/except mein
    catch ki ja sakti hai.
    
    Example:
        try:
            load_data("wrong_path.csv")
        except Exception as e:
            raise AeroGuardException(e, context="Loading flight header")
    """

    def __init__(self, error: Exception | str, context: str = ""):
        """
        Args:
            error   : Original exception ya error message string
            context : Kahan kya ho raha tha jab error aaya
                      Example: "Loading NGAFID flight header data"
        """
        # Error details extract karo
        if isinstance(error, Exception):
            details = get_error_details(error)
        else:
            # Agar string pass ki toh basic details banao
            details = {
                "file_name": "Unknown",
                "line_number": 0,
                "error_message": str(error),
                "error_type": "AeroGuardError"
            }

        # Human-readable error message banao
        self.error_message = (
            f"\n"
            f"{'='*60}\n"
            f"  AEROGUARD EXCEPTION\n"
            f"{'='*60}\n"
            f"  Type    : {details['error_type']}\n"
            f"  Message : {details['error_message']}\n"
            f"  File    : {details['file_name']}\n"
            f"  Line    : {details['line_number']}\n"
            f"  Context : {context if context else 'Not provided'}\n"
            f"{'='*60}"
        )

        # Logger se bhi log karo — file aur terminal dono mein jayega
        logger.error(self.error_message)

        # Parent Exception class ko initialize karo
        # Taaki yeh proper Exception ki tarah behave kare
        super().__init__(self.error_message)

    def __str__(self):
        """
        Jab bhi exception print hoga ya str() call hoga
        toh yeh formatted message dikhega
        """
        return self.error_message


# ============================================================
# Specific Exception Classes
# Har module ke liye alag exception — debugging mein aasani
# ============================================================

class DataIngestionException(AeroGuardException):
    """
    Data loading aur ingestion ke dauran aane wali errors.
    Example: NGAFID parquet files nahi mili, CSV corrupt hai
    """
    pass


class DataValidationException(AeroGuardException):
    """
    Data validation ke dauran aane wali errors.
    Example: Required columns missing hain, data types galat hain
    """
    pass


class DataTransformationException(AeroGuardException):
    """
    Feature engineering aur preprocessing ke dauran errors.
    Example: Normalization fail hua, NaN handle nahi hua
    """
    pass


class ModelTrainingException(AeroGuardException):
    """
    Model training ke dauran aane wali errors.
    Example: Insufficient data, convergence fail
    """
    pass


class ModelPredictionException(AeroGuardException):
    """
    Inference/prediction ke dauran aane wali errors.
    Example: Model file nahi mili, input shape galat hai
    """
    pass


class AnomalyDetectionException(AeroGuardException):
    """
    Anomaly detection ke dauran aane wali errors.
    Example: Autoencoder reconstruct nahi kar paya
    """
    pass


class AlertGenerationException(AeroGuardException):
    """
    Alert generate karne ke dauran aane wali errors.
    Example: XAI explanation fail hua, severity calculate nahi hua
    """
    pass


class ConfigurationException(AeroGuardException):
    """
    Config file load karne ya parse karne mein errors.
    Example: config.yaml missing hai, required key nahi mili
    """
    pass