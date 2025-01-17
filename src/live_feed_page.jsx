import React, { useState } from 'react';
import { TextField, Box, Button, Typography, Container } from '@mui/material';

const LiveFeedPage = () => {
    const [cameraURL, setCameraURL] = useState('');

    return (
        <Container maxWidth="md">
            <Box
                sx={{
                    marginTop: 8,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                }}
            >
                <Typography variant="h4" gutterBottom>
                    Live Camera Feed
                </Typography>
                <TextField
                    fullWidth
                    margin="normal"
                    label="Enter Camera URL"
                    variant="outlined"
                    value={cameraURL}
                    onChange={(e) => setCameraURL(e.target.value)}
                />
                {cameraURL && (
                    <Box sx={{ mt: 4, textAlign: 'center' }}>
                        <video
                            src={cameraURL}
                            controls
                            autoPlay
                            style={{ width: '100%', maxHeight: '500px' }}
                        >
                            Your browser does not support video playback.
                        </video>
                    </Box>
                )}
                <Button
                    variant="contained"
                    color="primary"
                    sx={{ mt: 3 }}
                    onClick={() => alert('Fetching live feed...')}
                >
                    Start Feed
                </Button>
            </Box>
        </Container>
    );
};

export default LiveFeedPage;